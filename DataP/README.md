# DataP — Preprocessed Relational Event Sequences

Preprocessed data for the Endogeneity project (Gibson 2021, *Social Networks*) and the Stability project (Gibson, in prep). Each file is an `.Rdata` object containing two objects:

| Object | Type | Description |
|--------|------|-------------|
| `edgelist` | `data.frame` | Ordered event sequence: rows are events, columns are (time, sender, receiver). Actors are re-coded to consecutive integers starting at 1. Self-loops are removed. |
| `n` | `integer` | Number of distinct actors in this network. |

Column naming in `edgelist` is not perfectly standardized across domains (some files use `V1`/`V2`/`V3`, others use named columns); the column *order* is always (time, sender, receiver).

The corpus spans ten social domains. **Finch networks are excluded from all analyses** — they contain non-human (zebra finch) interactions and serve only as a robustness reference.

---

## Domain Descriptions

### Classroom/ — `McFarland.Rdata`

Single network. McFarland classroom interaction data, loaded from the `networkDynamic` R package (`McFarland_cls33_10_16_96`). Records spoken utterances directed from one participant (student or teacher) to another during a single high-school class session. Group-addressed utterances are re-coded to a synthetic "group" node (node 21).

- **Networks:** 1
- **Events:** 691
- **Actors:** 21 (including synthetic group node)
- **Visibility:** High (interactions occur in a shared, observable space)
- **Valence:** Affiliative

---

### Email/ — `eu1.txt*.Rdata`, `eu2.txt*.Rdata`, `eu3.txt*.Rdata`, `eu4.txt*.Rdata`

Email communication from four departments of a European research institution (SNAP `email-Eu-core-temporal` dataset, Departments 1–4). Each department's event sequence is split into overlapping 5,001-event chunks (e.g., `eu1.txt1.Rdata` = events 1–5001, `eu1.txt5001.Rdata` = events 5001–10001, etc.). Actor IDs are re-coded to consecutive integers within each chunk, so `n` varies by chunk.

- **Networks:** 33 (across 4 departments, multiple chunks each)
- **Events per file:** 5,001
- **Actors per file:** ~46–55 (varies by chunk)
- **Source:** [SNAP email-Eu-core-temporal](https://snap.stanford.edu/data/email-Eu-core-temporal.html)
- **Visibility:** Low (email is private)
- **Valence:** Affiliative

---

### finch/ — `finch.Rdata`

⚠️ **Excluded from all analyses.** Non-human interaction network.

Zebra finch social interaction data, derived from `full_data_plus_cop_quest.tsv`. Records directed interactions (approaches, displacements) between individually marked birds. Included in `DataP` for reference but filtered out in all REM analyses via `!grepl("finch", ...)`.

- **Networks:** 1
- **Events:** 860
- **Actors:** 14

---

### Gangs/ — `gangnet1.Rdata` … `gangnet100.Rdata`

100 synthetic gang conflict interaction networks. Each network is an independent draw from a simulation model (`simgangget()` in `produce_gang_nets.R`) calibrated to real gang violence data. The simulation preserves the retaliatory and geographic structure of inter-gang conflict in an urban context.

- **Networks:** 100
- **Events per network:** 450 (fixed)
- **Actors per network:** 33 (fixed)
- **Visibility:** High (gang conflicts occur in public space and are observable to rival groups)
- **Valence:** Conflictual

---

### ftf/ — `network0.Rdata` … `network61.Rdata`

62 face-to-face glance interaction networks collected during structured group tasks. Each event records a directed visual attention act (a "glance") from one participant to another, extracted from video coding. Networks come from the Glances study and are indexed 0–61.

- **Networks:** 62
- **Events per network:** 10,000 (fixed window)
- **Actors per network:** 7–8 (varies by session)
- **Visibility:** Low (mutual gaze is visible to the dyad but glance direction is not publicly broadcast; treated as low-visibility in analyses)
- **Valence:** Affiliative

---

### PMs/ — `ucimsg1.Rdata`, `ucimsg7001.Rdata`, … `ucimsg49001.Rdata`

Private messages from the UCI College Messaging dataset (Panzarasa et al. 2009). The full sequence is split into 7,001-event chunks; actor IDs are re-coded within each chunk. Chunks are named by the starting event index.

- **Networks:** 8 chunks
- **Events per file:** 7,001
- **Actors per file:** ~600–760 (varies by chunk)
- **Source:** [UCI Network Data Repository — College Msg](http://konect.cc/networks/opsahl-ucsocial/)
- **Visibility:** Low (private messaging)
- **Valence:** Affiliative

---

### SMS/ — `sms5001.Rdata`, `sms10001.Rdata`, `sms15001.Rdata`

SMS (text message) communication sequences, split into 5,001-event chunks. Derived from a large mobile phone dataset; actor IDs are re-coded within each chunk.

- **Networks:** 3 chunks (representing 153 networks after within-chunk splitting in `Datadf`)
- **Events per file:** 5,001
- **Actors per file:** ~370–410 (varies by chunk)
- **Visibility:** Low (private messaging)
- **Valence:** Affiliative

---

### Twitter/ — `twitter_boston_relevent.rdata`

Single small Twitter network. Directed reply/mention interactions among users discussing an event in the Boston area, formatted for the `relevent` package. One of the EMO (emotion) Twitter networks used in Gibson (2021).

- **Networks:** 1
- **Events:** 227
- **Actors:** 20
- **Visibility:** High (Twitter interactions are public)
- **Valence:** Affiliative

---

### War/ — `conflict.Rdata`

International conflict event sequence derived from the Integrated Crisis Early Warning System (ICEWS) / Interstate Wars and Battles Dataset (IWBD 1.0). Each event is a directed military action from an attacker nation to a defender nation. Actor pool is capped at 200 nations; actors are re-coded to consecutive integers.

- **Networks:** 1
- **Events:** 1,708
- **Actors:** 200 (capped)
- **Source:** IWBD 1.0 (`iwbd1.0.csv`)
- **Visibility:** High (interstate conflict is publicly observable)
- **Valence:** Conflictual

---

### Phone/ — `Phone.Rdata`

Phone call communications from a military command-and-control exercise (TSOF dataset). Sender and receiver role names are re-coded to consecutive integers. A small number of self-loop calls are removed.

- **Networks:** 1
- **Events:** 196
- **Actors:** 94
- **Visibility:** High (in the DoD exercise context, communication roles and patterns are organizationally visible)
- **Valence:** Affiliative

---

## Domain Summary Table

| Directory | Domain label | Networks | Visibility | Valence |
|-----------|-------------|----------|------------|---------|
| `Classroom/` | Classroom interactions | 1 | High | Affiliative |
| `Email/` | Emails (EU departments) | 33 | Low | Affiliative |
| `finch/` | Finch (excluded) | 1 | — | — |
| `ftf/` | Glance networks | 62 | Low | Affiliative |
| `Gangs/` | Gang violence | 100 | High | Conflictual |
| `PMs/` | Private messages (UCI) | 8 | Low | Affiliative |
| `SMS/` | SMS messages | 3 | Low | Affiliative |
| `Twitter/` | Twitter (EMO) | 1 | High | Affiliative |
| `War/` | International conflict | 1 | High | Conflictual |
| `Phone/` | Phone calls (DoD) | 1 | High | Affiliative |

---

## Notes on Processing

- All files were produced by `build_datasets.R` in the project root.
- Actor IDs within each file are consecutive integers starting at 1 and are **not** comparable across files (even within the same domain for chunked datasets).
- For the REM analyses in the Stability project, these files are **not** used directly. A further processing step computes all endogenous statistics and produces `poss_edges` data frames, which are stored in `../Datadf/`. See `greedy_opt.R` and `simple_study.R` in the Stability project.
- The `finch/` directory is retained for completeness but is excluded from all analyses via `datafiles <- datafiles[!grepl("finch", datafiles, ignore.case = TRUE)]`.

---

## References

- Gibson, C. B. (2021). Endogeneity in relational event sequences. *Social Networks*, 68, 276–287.
- McFarland, D. A. (2001). Student resistance: How the formal and informal organization of classrooms facilitate everyday acts of defiance. *American Journal of Sociology*, 107(3), 612–678.
- Panzarasa, P., Opsahl, T., & Carley, K. M. (2009). Patterns and dynamics of users' behavior and interaction: Network analysis of an online community. *JASIST*, 60(5), 911–932.
