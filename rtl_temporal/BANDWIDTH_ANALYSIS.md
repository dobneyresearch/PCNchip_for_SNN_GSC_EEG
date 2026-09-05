# Router bandwidth for per-timestep credit (audit refinement #2, 2026-08-20)

**Question.** Temporal credit routes the adjoint δ **per timestep** (×T), where the static chip routed
error **per sample**. Does the Part II router (fixed-divide gather, best-effort error, 4-bit `frame_seq`)
have the bandwidth, and does its tagging accommodate the temporal window?

**Facts.** T = 200 timesteps/sample; topology (20, 256, 256, 256), 35 classes; δ message = **6-bit**
(`delta_bits`, post `route_quant` #1); forward spike = **1-bit**; forward activation = 8-bit (ADC).
Router: FIXED-DIVIDE gather (÷ nominal fan-in, no wait-for-all barrier — a missing partial just
under-drives), best-effort error channel, `frame_seq[3:0]` = **16 frames in flight**.

## 1. Per-timestep traffic (network-wide, dense worst case)
| direction | messages / timestep | bits / msg | bits / timestep |
|---|---|---|---|
| forward spikes (L1,L2,L3 out) | 256·3 = 768 (dense; ~20% firing ⇒ ~150) | 1 | ~768 |
| backward δ (RO→L3, L3→L2, L2→L1) | 256·3 = 768 | 6 | ~4,608 |
| **total** | **~1,536 msg** | | **~5.4 Kbit** |

Per sample (T=200): ~1.08 Mbit; ~307k messages.

## 2. The ×T factor is absorbed by the timestep DURATION
The ×200 message-rate increase does **not** raise the absolute rate to anything demanding, because the
T timesteps span the whole sample. For a 1 s keyword sample (T=200 ⇒ **5 ms/timestep**):
- aggregate **~1.1 Mbit/s** network-wide (~5.4 Mbit/s even at a fast 1 ms/timestep);
- **~307k messages/s** ⇒ at a modest **100 MHz** router that is **~326 cycles of budget per message**.

A single 8-bit link at 100 MHz carries 800 Mbit/s, so the entire network's per-timestep traffic is **<1%**
of one link — **2–3 orders of magnitude of headroom.** Per-timestep δ routing is **not** a bandwidth
bottleneck; the bottleneck (if any) is compute/latency, not the interconnect.

## 3. Router mechanisms still hold for per-timestep δ
- **Best-effort + commutative gather.** δ partials are order-independent (fixed-divide sums them) and
  droppable — a lost/late δ under-fills a jug that self-corrects over later sweeps (eq. 6). So the
  higher-frequency δ traffic needs **no tighter delivery** than the static case; it is the same
  best-effort channel, just more often.
- **Forward reliable.** Forward spikes still need reliable delivery (a lost spike corrupts the pass), but
  they are 1-bit and sparse — trivial volume on the reliable channel.
- **Bidirectional, pipelined.** Forward (timestep t) and backward (δ for timestep t−N, the FIFO learning
  delay) share the same physical links offset by the window latency — the same bidirectional packet fabric
  as Part II, naturally pipelined.

## 4. The one thing to size-check: `frame_seq` depth
Static used few frames in flight; temporal has ~N timesteps concurrently tagged (forward at t, backward at
t−N). With **N≈8**, the existing **4-bit `frame_seq` (16 frames) covers it with 2× margin.** ✅ Fits. Only
if the window or forward/backward pipeline depth exceeds ~16 timesteps would a **5-bit** field be needed —
worth widening `frame_seq` if a deeper temporal buffer is ever adopted, but not at N≈8.

## Conclusion
Per-timestep credit routing fits the Part II router comfortably: the ×T message-rate rise is absorbed by
ms-scale timesteps (2–3 orders of magnitude of link headroom), the best-effort/commutative/fixed-divide
mechanisms carry δ unchanged (just more frequently), and the 4-bit `frame_seq` accommodates the N≈8
in-flight window with margin. **No router redesign is required; the single parameter to revisit for a
deeper temporal window is the `frame_seq` width.**
