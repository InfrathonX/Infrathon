# Architecture and experiment ledger

## Active public candidate: v3.11 late owner exchange

Gateway SHA-256: `d27e8cff1fff5d22de87ea34af905d3bca74c7bd9527bd3c478a113a89d601b8`.
V3.11 returns to the exact high-locality V3.8 borrower policy and composes two
separately measured mechanisms.  V3.10's global-oldest ordinal-four-or-later
selector at 60 seconds found the tail, but its expanded cold circulation lost
rate and locality.  V3.5's completion-safe one-owner donation kept rate and
locality, but its ordinal-one selector did not move the recurring tail.  V3.11
applies the late selector to a one-for-one owner transfer and adds no
late-triggered expanded borrower entitlement.

## Program state machine

```text
global chronological head, ordinal>=4, age>=60 s
  ordinary 32/292 + hard R fits -> exact V3.8 work conservation; no forced exchange
  ordinary lane or hard R blocks -> EXCHANGE_PENDING

EXCHANGE_PENDING + free owner envelope + hard fit -> PINNED direct promotion
EXCHANGE_PENDING + compatible completed donor     -> atomic owner exchange
EXCHANGE_PENDING + no safe donor/budget           -> exact V3.8 ordinary wait

target 2xx success -> ACTING + PINNED
target error/cancel -> prior target boundary; unearned pin removed
donor               -> remains safely PAUSED; never resurrected speculatively
```

The target must be the sole global chronological FIFO head, identified and
non-ephemeral, unpinned and inactive, neither cancelled nor in flight nor
release-pending, ordinal at least four, queued at least 60 seconds, and
representable in the owner envelope.  A donor must be PINNED and ACTIVE at a completed-response
boundary, have at least one successful owner turn, and have no queued/inflight
work or pending release.  Stale donors are preferred; if none exists, the
oldest completion-safe donor is a lease-breaking fallback justified by the
target's full-turnover-window debt.  Each real dispatch wake transfers at most
one slot.  While that exact blocked debt exists, genuinely free handoff
promotion remains work-conserving, but a victim-consuming borrower handoff
defers so it cannot steal the target's bounded turnover opportunity.

## Resource model

All preflight and mutation are lock-held with no await:

```text
R' = R - actual_resident(donor) + request_increment(target) <= 1188
C' = C - owner_envelope(donor) + desired(target)            <= 896
owner count after a one-for-one transfer                    <= 128
rolling transfers in 60 s                                  <= 32 programs
rolling transferred envelopes in 60 s                      <= 224 pages
ordinary BORROWER                                           <= 32 / 292
shared BORROWER (unchanged V3.8 debt/underfill; no late grant) <= 64 / 448
upstream requests                                           <= 192
```

`_demote_pin(..., rotation=True)` records turnover before `_promote_pin`
marks the already-queued target; the normal admission loop then installs its
full PINNED request claim before its one handler opens upstream.  An exchanged
target never enters the borrower count/page totals.  A crossed-deadline race
found after immutable V3.10 is also corrected: rearming preserves an existing
live due handle, and sub-microsecond future deadlines are scheduled rather
than dropped.  A timer only re-enters dispatch and cannot create model work.
Transport, estimation, response observation, release, and bounded cleanup are
otherwise byte-for-byte V3.8.

## Online monitoring

The active V3.11 gateway derives its control state only from traffic it
legitimately proxies: per-program request ordinal, enqueue age, in-flight and
completion boundaries, observed token usage, and bounded rolling transfer
counts/page envelopes. No hidden workload identity or backend-private state is
used. The chronological head and timer wakeups expose starvation pressure to
the admission loop without creating model requests themselves.

## Admission, scheduling, and restoration policy

The lock-held invariants and `EXCHANGE_PENDING` transition above are the
admission policy. Scheduling is global chronological FIFO at the late boundary,
with at most one safe owner exchange per dispatch wake. A target earns its pin
only after a successful response; error or cancellation restores its prior
boundary and removes an unearned pin. Donors remain paused at a completed
response boundary, while all request claims, lane counters, timer handles, and
turnover budgets have bounded cleanup paths.

## Experiment ledger

| Candidate / plane | Code hash | Architectural change | Causal hypothesis | Expected evidence | Reject if | Result / decision |
|---|---|---|---|---|---|---|

## Protected best

The installed starter is the V3.11 gateway (`d27e8cff...`). Treat it as the
rollback checkpoint until a replacement has its own evidence.
