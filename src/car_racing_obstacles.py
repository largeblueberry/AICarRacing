"""CarRacing-v3 variant with random static obstacles on the road.

The agent must learn to steer around obstacles. Design notes:

* Obstacles are real Box2D static bodies placed on random track tiles, so the
  car physically collides with them (bounces / loses speed).
* Each obstacle quad is appended to ``self.road_poly`` so it is DRAWN into the
  rendered frame. This matters because the agent's input is pixels: the
  obstacle must be visible in the 96x96 observation to be avoidable. White is
  used for strong contrast in grayscale (road ~102, grass ~162, obstacle 255).
* Collisions are detected via a FrictionDetector subclass; each step in which
  a new car<->obstacle contact begins subtracts ``obstacle_penalty`` from the
  reward (episode continues — penalty-only design).
* Obstacle placement uses ``self.np_random`` so it is reproducible per seed.

Usage:
    import src.car_racing_obstacles  # registers the env id
    env = gym.make("CarRacingObstacles-v0", continuous=True,
                   n_obstacles=10, obstacle_penalty=15.0)
"""

import math

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector, TRACK_WIDTH

try:
    from Box2D.b2 import fixtureDef, polygonShape
except ImportError as e:
    raise ImportError(
        "Box2D is required for CarRacingObstacles "
        '(`pip install swig` then `pip install "gymnasium[box2d]"`)'
    ) from e

# White: grayscale 255 vs road ~102 and grass ~162 -> clearly visible to the agent.
OBSTACLE_COLOR = (255, 255, 255)


class _ObstacleMarker:
    """userData marker so the contact listener can recognize obstacle bodies."""

    def __init__(self):
        self.is_obstacle = True


class ObstacleFrictionDetector(FrictionDetector):
    """FrictionDetector that additionally reports car<->obstacle contacts."""

    def BeginContact(self, contact):
        if self._is_obstacle_contact(contact):
            self.env.obstacle_hits_pending += 1
            return
        super().BeginContact(contact)

    def EndContact(self, contact):
        if self._is_obstacle_contact(contact):
            return
        super().EndContact(contact)

    @staticmethod
    def _is_obstacle_contact(contact) -> bool:
        try:
            for ud in (contact.fixtureA.body.userData, contact.fixtureB.body.userData):
                if ud is not None and getattr(ud, "is_obstacle", False):
                    return True
        except Exception:
            # Belt-and-braces: never let a torn-down contact crash the sim.
            return True
        return False


class CarRacingObstacles(CarRacing):
    """CarRacing with ``n_obstacles`` random static obstacles on the road.

    Args (in addition to CarRacing's):
        n_obstacles: number of obstacles to scatter on the track.
        obstacle_penalty: reward subtracted on each step where a new
            car<->obstacle contact begins (episode continues).
        obstacle_size: obstacle width as a fraction of the road half-width
            (TRACK_WIDTH). 0.4 -> ~2.7 units wide vs ~13.3 road width, so
            there is always room to pass on at least one side.
        start_clear_tiles: number of tiles after the start line kept free of
            obstacles so the car is never spawned into a wall.
        min_tile_gap: minimum tile-index distance between obstacles.
    """

    def __init__(self, *args, n_obstacles: int = 10, obstacle_penalty: float = 15.0,
                 obstacle_size_min: float = 0.25, obstacle_size_max: float = 0.6,
                 start_clear_tiles: int = 30, min_tile_gap: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_obstacles = n_obstacles
        self.obstacle_penalty = obstacle_penalty
        # Each obstacle's side is sampled per-obstacle (per-reset, via np_random
        # so it's seed-reproducible) from [min, max] as a fraction of TRACK_WIDTH.
        # Set min == max for a fixed size. max=0.6 -> full width ~0.6*TRACK_WIDTH,
        # ~30% of the road, so a drivable gap always remains.
        self.obstacle_size_min = obstacle_size_min
        self.obstacle_size_max = obstacle_size_max
        self.start_clear_tiles = start_clear_tiles
        self.min_tile_gap = min_tile_gap
        self.obstacle_bodies = []
        self.obstacle_sizes = []  # sampled half-extents, for inspection/debug
        self.obstacle_hits_pending = 0

    # --- lifecycle -------------------------------------------------------
    def _destroy(self):
        # CRITICAL: detach the contact listener before destroying bodies.
        # If the car is touching an obstacle when reset() is called, Box2D
        # fires EndContact callbacks DURING DestroyBody; re-entering Python
        # on a half-destroyed body segfaults box2d-py (observed on Linux).
        # The listener is reinstalled by CarRacing.reset() right after this.
        self.world.contactListener = None
        self.world.contactListener_bug_workaround = None
        for body in self.obstacle_bodies:
            self.world.DestroyBody(body)
        self.obstacle_bodies = []
        super()._destroy()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # super().reset() installed a fresh FrictionDetector; swap in ours so
        # obstacle contacts are detected (tile/lap logic is inherited).
        detector = ObstacleFrictionDetector(self, self.lap_complete_percent)
        self.world.contactListener_bug_workaround = detector
        self.world.contactListener = detector

        self.obstacle_hits_pending = 0
        self._spawn_obstacles()

        # Refresh the first observation so obstacles are visible from frame 0
        # (super().reset() rendered before obstacles existed).
        obs = self._render("state_pixels")
        self.state = obs
        return obs, info

    def step(self, action):
        obs, step_reward, terminated, truncated, info = super().step(action)

        hits = self.obstacle_hits_pending
        self.obstacle_hits_pending = 0
        # One penalty per step (a single crash can begin hull+wheel contacts
        # simultaneously; counting them all would over-penalize).
        if action is not None and hits > 0:
            step_reward -= self.obstacle_penalty
            info["obstacle_hit"] = True
        info["obstacle_hits"] = hits
        return obs, step_reward, terminated, truncated, info

    # --- obstacle placement ----------------------------------------------
    def _spawn_obstacles(self):
        self.obstacle_bodies = []
        n_tiles = len(self.track)
        # The track is a LOOP: tiles near n_tiles-1 sit right behind the start
        # line, so keep BOTH ends clear or an obstacle can spawn on the car.
        if n_tiles <= 2 * self.start_clear_tiles + 1:
            return

        candidates = list(range(self.start_clear_tiles, n_tiles - self.start_clear_tiles))
        self.np_random.shuffle(candidates)

        chosen = []
        for idx in candidates:
            if len(chosen) >= self.n_obstacles:
                break
            if all(abs(idx - c) >= self.min_tile_gap for c in chosen):
                chosen.append(idx)

        self.obstacle_tile_indices = list(chosen)  # for debugging/inspection
        self.obstacle_sizes = []

        for idx in chosen:
            _, beta, x, y = self.track[idx]
            # Per-obstacle random size (seed-reproducible via np_random).
            size_frac = self.np_random.uniform(self.obstacle_size_min, self.obstacle_size_max)
            half = size_frac * TRACK_WIDTH / 2.0
            self.obstacle_sizes.append(half)
            # Keep the obstacle's outer edge inside ~80% of the road half-width so
            # the agent always has a drivable gap on at least one side. Clamp >= 0
            # in case a (mis)configured max would exceed the bound.
            max_offset = max(0.0, 0.8 * TRACK_WIDTH - half)
            offset = self.np_random.uniform(-max_offset, max_offset)
            # (cos(beta), sin(beta)) is the lateral (across-road) axis, same
            # convention _create_track uses for the road edge vertices.
            px = x + offset * math.cos(beta)
            py = y + offset * math.sin(beta)

            body = self.world.CreateStaticBody(
                position=(px, py),
                angle=beta,
                fixtures=fixtureDef(shape=polygonShape(box=(half, half))),
            )
            body.userData = _ObstacleMarker()
            self.obstacle_bodies.append(body)

            # Draw the obstacle into the rendered frame (and thus the agent's
            # pixel observation) as a rotated quad.
            c, s = math.cos(beta), math.sin(beta)
            quad = [
                (px + dx * c - dy * s, py + dx * s + dy * c)
                for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))
            ]
            self.road_poly.append((quad, OBSTACLE_COLOR))


# Register so gym.make("CarRacingObstacles-v0", ...) works anywhere the
# project root is importable. Guarded against double-registration.
if "CarRacingObstacles-v0" not in gym.registry:
    gym.register(
        id="CarRacingObstacles-v0",
        entry_point="src.car_racing_obstacles:CarRacingObstacles",
        max_episode_steps=1000,
        reward_threshold=900,
    )
