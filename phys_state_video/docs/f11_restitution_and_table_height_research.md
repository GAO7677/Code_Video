# F11 Restitution and Table-Height Research

## Scope

This note evaluates how to make the F11 table roll-off height contrast exceed 25 cm
without changing the floor restitution. The target comparison is the first ground-bounce
apex for the 0.46 m low table and the high-table case.

## What a restitution value represents

Restitution is a contact-pair property, not a universal property of a floor. Ball
construction, internal pressure, impact speed, temperature, humidity, and the contact
surface all affect the observed rebound.

Relevant primary sources:

- The ITF specifies tennis-ball rebound against a smooth, rigid, horizontal high-mass
  block. Its test drops a ball from 254 cm and measures the rebound height. The
  approved Type 1/2/3/high-altitude ranges are 138-151 cm, 135-147 cm, 135-147 cm,
  and 122-135 cm respectively. These correspond to effective normal restitution
  ranges of about 0.737-0.771, 0.729-0.761, 0.729-0.761, and 0.693-0.729.
  Sources: [ITF Technical Booklet, ball specifications](https://www.itftennis.com/media/15648/2026-technical-booklet.pdf#page=8)
  and [ITF rebound test method](https://www.itftennis.com/media/15648/2026-technical-booklet.pdf#page=14).
- The ITF also states that the rubber core, internal gas, temperature, humidity, and
  atmospheric pressure affect a tennis ball's properties. This is why a single
  material-only "normal floor coefficient" is not defensible.
  Source: [ITF Technical Booklet, controlled environment](https://www.itftennis.com/media/15648/2026-technical-booklet.pdf#page=11).
- FIBA specifies basketball-floor ball-rebound performance separately by flooring
  system: at least 93% for Level 1 wood/glass flooring and at least 90% for Level 2
  synthetic flooring under EN 12235. These are test-system requirements, not a
  direct Bullet restitution value.
  Source: [FIBA Official Basketball Rules 2024, floor requirements](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf#page=18).

## Consequences for the current simulator

Bullet combines the two body restitution values by multiplication. See
[`btManifoldResult::calculateCombinedRestitution`](https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionDispatch/btManifoldResult.cpp).

The current F11 setup is:

| Parameter | Value |
| --- | ---: |
| Ball mass | 2.50 kg |
| Ball restitution | 1.00 |
| Floor restitution | 0.62 |
| Effective Bullet contact restitution | 0.62 |
| Low table height | 0.46 m |

Therefore, increasing ball mass does not change the free-fall acceleration or the
static-floor rebound. Increasing ball restitution is also not a physical option here:
the ball is already at 1.00, so any increase would imply energy gain, while any
decrease only reduces the effective contact restitution.

## Deterministic PyBullet checks

The following values were obtained with the project F11 geometry, gravity, speed,
friction, ball parameters, and floor restitution held fixed. The simulation used the
same 240 Hz solver settings as the renderer and selected the first floor-contact
rebound.

### Mass sweep with floor restitution fixed at 0.62

| Ball mass (kg) | Low-table bounce (cm) | High-table bounce (cm) | Difference (cm) |
| ---: | ---: | ---: | ---: |
| 0.25 | 19.14 | 40.13 | 20.99 |
| 0.50 | 19.14 | 40.13 | 20.99 |
| 1.00 | 19.14 | 40.13 | 20.99 |
| 2.50 | 19.14 | 40.13 | 20.99 |
| 5.00 | 19.14 | 40.13 | 20.99 |
| 10.00 | 19.14 | 40.13 | 20.99 |
| 25.00 | 19.14 | 40.13 | 20.99 |

### High-table sweep with floor restitution fixed at 0.62

The current generator allows table heights up to 1.02 m. With the low table fixed at
0.46 m, a 1.02 m high table produces a 29.87 cm high-minus-low first-bounce contrast
in the deterministic test, exceeding the 25 cm target without changing the floor or
ball parameters.

## Recommendation

Keep the floor restitution at 0.62 and retain one fixed ball specification across all
F11 cases. Use 0.46 m and 1.02 m as the contrast pair, then render and validate that
the ball and rebound remain inside the camera frame. Do not use mass as a proxy for
bounciness against a static floor.
