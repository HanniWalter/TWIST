# Vergleich: Student vs. Future Student Input-Vektoren

## Übersicht

| Eigenschaft | `K1MimicStuRLCfg` (Original) | `K1MimicStuRLCfg_future` | `K1MimicStuRLCfg_future_keypoints` |
|-------------|------------------------------|--------------------------|------------------------------------|
| `obs_type` | `'student'` | `'student_future_no_keypoints'` | `'student_future'` |
| `history_len` | 10 | 0 | 0 |
| `n_obs_single` | 93 (mimic + proprio) | 625 (future_mimic + proprio) | 1165 (priv_mimic + proprio) |
| `num_observations` | 1023 | 625 | 1165 |
| Key Body Positions | ❌ | ❌ | ✅ |

---

## Original Student (`K1MimicStuRLCfg`)

### Observation Struktur
```
obs_type = 'student'
history_len = 10

n_mimic_obs = 8 + 20 = 28        # Einzelner Zeitschritt
n_proprio = 3 + 2 + 3*20 = 65    # gravity(3) + commands(2) + dof_pos/vel/actions(60)

n_obs_single = n_mimic_obs + n_proprio = 28 + 65 = 93
num_observations = n_obs_single * (history_len + 1) = 93 * 11 = 1023
```

### Input-Vektor Layout (1023 Elemente)
```
Index 0-92:     Aktueller Zeitschritt (obs_buf)
  [0-27]:       mimic_obs (motion first)
    [0-3]:      target_root_quat (4)
    [4-6]:      target_root_pos (3)  
    [7]:        root_height (1)
    [8-27]:     target_dof_pos (20)
  [28-92]:      proprio_obs (65)
    [28-30]:    projected_gravity (3)
    [31-32]:    commands (2)
    [33-52]:    dof_pos (20)
    [53-72]:    dof_vel (20)
    [73-92]:    actions (20)

Index 93-185:   History Step 1 (t-1)
Index 186-278:  History Step 2 (t-2)
...
Index 930-1022: History Step 10 (t-10)
```

### Zusammenfassung
- **Motion-Daten:** Nur aktueller Zeitschritt (28 Features)
- **History:** 10 vergangene Zeitschritte (temporal context)
- **Gesamt:** 93 × 11 = 1023 Elemente

---

## Future Student mit Keypoints (`K1MimicStuRLCfg_future_keypoints`)

### Observation Struktur
```
obs_type = 'student_future'
history_len = 0

n_priv_mimic_obs = 20 * (8 + 20 + 27) = 20 * 55 = 1100  # 20 Zukunfts-Schritte
n_proprio = 3 + 2 + 3*20 = 65

n_obs_single = n_priv_mimic_obs + n_proprio = 1100 + 65 = 1165
num_observations = n_obs_single = 1165  # Keine History-Multiplikation
```

### Input-Vektor Layout (1165 Elemente)
```
Index 0-1099:   priv_mimic_obs (motion first - 20 Zukunfts-Schritte)
  Für jeden Schritt i (0-19), tar_obs_steps = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
    [i*55 + 0-3]:    target_root_quat (4)
    [i*55 + 4-6]:    target_root_pos (3)
    [i*55 + 7]:      root_height (1)
    [i*55 + 8-27]:   target_dof_pos (20)
    [i*55 + 28-54]:  key_body_pos (9 bodies × 3 = 27)

  Aufschlüsselung:
    Index 0-54:     Zukunft t+1
    Index 55-109:   Zukunft t+5
    Index 110-164:  Zukunft t+10
    Index 165-219:  Zukunft t+15
    Index 220-274:  Zukunft t+20
    Index 275-329:  Zukunft t+25
    Index 330-384:  Zukunft t+30
    Index 385-439:  Zukunft t+35
    Index 440-494:  Zukunft t+40
    Index 495-549:  Zukunft t+45
    Index 550-604:  Zukunft t+50
    Index 605-659:  Zukunft t+55
    Index 660-714:  Zukunft t+60
    Index 715-769:  Zukunft t+65
    Index 770-824:  Zukunft t+70
    Index 825-879:  Zukunft t+75
    Index 880-934:  Zukunft t+80
    Index 935-989:  Zukunft t+85
    Index 990-1044: Zukunft t+90
    Index 1045-1099: Zukunft t+95

Index 1100-1164: proprio_obs (65)
    [1100-1102]: projected_gravity (3)
    [1103-1104]: commands (2)
    [1105-1124]: dof_pos (20)
    [1125-1144]: dof_vel (20)
    [1145-1164]: actions (20)
```

### Zusammenfassung
- **Motion-Daten:** 20 Zukunfts-Schritte (1100 Features)
- **History:** Keine (0 vergangene Zeitschritte)
- **Gesamt:** 1100 + 65 = 1165 Elemente

---

## Future Student (`K1MimicStuRLCfg_future`) - Standard

### Observation Struktur
```
obs_type = 'student_future_no_keypoints'
history_len = 0

n_future_mimic_obs = 20 * 28 = 560  # 20 Zukunfts-Schritte OHNE key_body_pos
n_proprio = 3 + 2 + 3*20 = 65

n_obs_single = n_future_mimic_obs + n_proprio = 560 + 65 = 625
num_observations = n_obs_single = 625
```

### Input-Vektor Layout (625 Elemente)
```
Index 0-559:   mimic_obs_all_steps (motion first - 20 Zukunfts-Schritte OHNE keypoints)
  Für jeden Schritt i (0-19), tar_obs_steps = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
    [i*28 + 0]:      root_height (1)
    [i*28 + 1-3]:    roll, pitch, yaw (3)
    [i*28 + 4-6]:    root_vel (3)
    [i*28 + 7]:      yaw_vel (1)
    [i*28 + 8-27]:   target_dof_pos (20)
    
  Aufschlüsselung:
    Index 0-27:     Zukunft t+1
    Index 28-55:    Zukunft t+5
    Index 56-83:    Zukunft t+10
    Index 84-111:   Zukunft t+15
    Index 112-139:  Zukunft t+20
    Index 140-167:  Zukunft t+25
    Index 168-195:  Zukunft t+30
    Index 196-223:  Zukunft t+35
    Index 224-251:  Zukunft t+40
    Index 252-279:  Zukunft t+45
    Index 280-307:  Zukunft t+50
    Index 308-335:  Zukunft t+55
    Index 336-363:  Zukunft t+60
    Index 364-391:  Zukunft t+65
    Index 392-419:  Zukunft t+70
    Index 420-447:  Zukunft t+75
    Index 448-475:  Zukunft t+80
    Index 476-503:  Zukunft t+85
    Index 504-531:  Zukunft t+90
    Index 532-559:  Zukunft t+95

Index 560-624: proprio_obs (65)
    [560-562]:   projected_gravity (3)
    [563-564]:   commands (2)
    [565-584]:   dof_pos (20)
    [585-604]:   dof_vel (20)
    [605-624]:   actions (20)
```

### Zusammenfassung
- **Motion-Daten:** 20 Zukunfts-Schritte (560 Features, OHNE key_body_pos)
- **History:** Keine (0 vergangene Zeitschritte)
- **Key Body Positions:** ❌ Nicht enthalten
- **Gesamt:** 560 + 65 = 625 Elemente

---

## Hauptunterschiede

| Aspekt | Original Student | Future Student | Future (keypoints) |
|--------|------------------|----------------|--------------------|
| **Temporal Focus** | Vergangenheit (History) | Zukunft (Future Targets) | Zukunft (Future Targets) |
| **Motion Features pro Step** | 28 (ohne key_body_pos) | 28 (ohne key_body_pos) | 55 (mit key_body_pos) |
| **Anzahl Zeitschritte** | 1 aktuell + 10 History = 11 | 20 Zukunft | 20 Zukunft |
| **Reihenfolge** | `[mimic, proprio] × 11` | `[mimic_all, proprio]` | `[priv_mimic, proprio]` |
| **Key Body Positions** | ❌ Nicht enthalten | ❌ Nicht enthalten | ✅ Enthalten (27 pro Step) |
| **Gesamtgröße** | 1023 | 625 | 1165 |

---

## Actor Model Erwartungen

Der `ActorCriticMimic` Actor erwartet:
```python
motion_obs = obs[:, :num_motion_observations]  # Erste N Elemente = Motion
proprio_obs = obs[:, num_motion_observations:]  # Rest = Proprio
```

### Für Original Student:
- `num_motion_observations = n_mimic_obs = 28`
- `num_motion_steps = 1`
- Actor sliced: `obs[:, :28]` → mimic_obs ✓

### Für Modified Student:
- `num_motion_observations = n_priv_mimic_obs = 1100`
- `num_motion_steps = 20`
- Actor sliced: `obs[:, :1100]` → priv_mimic_obs ✓

---

## Privileged Observations (Teacher)

Beide Configs haben identische `num_privileged_obs`:
```
n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
                  = 1100 + 65 + 72 = 1237

n_priv_info = 3 + 1 + 27 + 2 + 4 + 1 + 40 = 78
  - base_lin_vel: 3
  - root_height: 1  
  - key_body_pos: 27 (9 bodies × 3)
  - contact_mask: 2
  - priv_latent: 4
  - ?: 1
  - motor_strength: 40 (2 × 20)
```

Der Teacher verwendet die vollen privilegierten Observations mit zusätzlichen Informationen (base velocity, contacts, etc.), die der Student nicht beobachten kann.
