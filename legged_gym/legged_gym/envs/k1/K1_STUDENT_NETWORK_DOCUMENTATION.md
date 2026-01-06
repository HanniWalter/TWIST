# K1 Student Network (Modified) - Input/Output Dokumentation

Diese Dokumentation beschreibt die Eingabe- und Ausgabedimensionen des **modifizierten** Student-Netzwerks für den Booster K1 Roboter.

## Änderung gegenüber Original-Student

| Aspekt | Original Student | Modified Student |
|--------|------------------|------------------|
| **Zeitliche Info** | 10 vergangene Frames (Historie) | 20 zukünftige Motion-Targets |
| **Input Dimension** | 1023 | 1165 |
| **Konzept** | Zustandsschätzung aus Vergangenheit | Zielplanung aus Zukunft |

---

## Übersicht

| Parameter | Wert |
|-----------|------|
| **Input Dimension** | 1165 |
| **Output Dimension** | 20 |
| **Zukünftige Schritte** | 20 |
| **History Length** | 0 (keine) |

---

## Input: Observations (1165 Dimensionen)

Die Observation besteht aus **proprioceptiven Daten** und **zukünftigen Motion-Targets**:

```
num_observations = n_proprio + n_priv_mimic_obs
                 = 65 + 1100 = 1165
```

### 1. Proprioceptive Observations (`n_proprio` = 65)

| Index | Dimension | Beschreibung |
|-------|-----------|--------------|
| 0-2   | 3 | Projizierter Gravitationsvektor (in Body-Frame) |
| 3-4   | 2 | Commands (z.B. Geschwindigkeitsbefehle) |
| 5-24  | 20 | Aktuelle DOF-Positionen |
| 25-44 | 20 | Aktuelle DOF-Geschwindigkeiten |
| 45-64 | 20 | Letzte Aktionen |

```
n_proprio = 3 + 2 + 3 * num_actions
          = 3 + 2 + 3 * 20 = 65
```

### 2. Zukünftige Motion-Targets (`n_priv_mimic_obs` = 1100)

Für **20 zukünftige Zeitschritte** werden Motion-Targets bereitgestellt:

```python
tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
```

**Pro Zeitschritt (55 Dimensionen):**

| Index | Dimension | Beschreibung |
|-------|-----------|--------------|
| 0-3   | 4 | Ziel-Orientierung (Quaternion: w, x, y, z) |
| 4-6   | 3 | Ziel-Position (x, y, z) relativ zum Roboter |
| 7     | 1 | Zusätzliche Mimic-Info |
| 8-27  | 20 | Ziel-DOF-Positionen |
| 28-54 | 27 | Key Body Positionen (9 Bodies × 3D) |

```
n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9)
                 = 20 * (8 + 20 + 27) = 20 * 55 = 1100
```

### Key Bodies (9 Stück)

| Nr | Body Name | Beschreibung |
|----|-----------|--------------|
| 1 | `left_hand_end_ball` | Linke Hand |
| 2 | `right_hand_end_ball` | Rechte Hand |
| 3 | `left_foot_link` | Linker Fuß |
| 4 | `right_foot_link` | Rechter Fuß |
| 5 | `right_outer_toe_link` | Rechter äußerer Zeh |
| 6 | `left_outer_toe_link` | Linker äußerer Zeh |
| 7 | `right_inner_toe_link` | Rechter innerer Zeh |
| 8 | `left_inner_toe_link` | Linker innerer Zeh |
| 9 | `Head_2` | Kopf |

### Gesamtstruktur der Observation

```
Observation[1165] = [
    # Proprioception (65)
    gravity[3],                    # Index 0-2
    commands[2],                   # Index 3-4
    dof_pos[20],                   # Index 5-24
    dof_vel[20],                   # Index 25-44
    last_actions[20],              # Index 45-64
    
    # Zukünftige Motion-Targets (1100)
    # Schritt +1 Frame
    target_root_pose[8],           # Index 65-72
    target_dof_pos[20],            # Index 73-92
    target_key_body_pos[27],       # Index 93-119
    
    # Schritt +5 Frames
    target_root_pose[8],           # Index 120-127
    target_dof_pos[20],            # Index 128-147
    target_key_body_pos[27],       # Index 148-174
    
    # ... (weitere 18 Zeitschritte bis +95 Frames)
]
```

---

## Output: Actions (20 Dimensionen)

Die 20 Ausgabedimensionen entsprechen den Ziel-Gelenkpositionen für alle kontrollierten Gelenke:

### Gelenkzuordnung

| Index | Gelenk | Typ |
|-------|--------|-----|
| **Linker Arm** |
| 0 | ALeft_Shoulder_Pitch | Schulter Pitch |
| 1 | Left_Shoulder_Roll | Schulter Roll |
| 2 | Left_Elbow_Pitch | Ellbogen Pitch |
| 3 | Left_Elbow_Yaw | Ellbogen Yaw |
| **Rechter Arm** |
| 4 | ARight_Shoulder_Pitch | Schulter Pitch |
| 5 | Right_Shoulder_Roll | Schulter Roll |
| 6 | Right_Elbow_Pitch | Ellbogen Pitch |
| 7 | Right_Elbow_Yaw | Ellbogen Yaw |
| **Linkes Bein** |
| 8 | Left_Hip_Yaw | Hüfte Yaw |
| 9 | Left_Hip_Roll | Hüfte Roll |
| 10 | Left_Hip_Pitch | Hüfte Pitch |
| 11 | Left_Knee_Pitch | Knie Pitch |
| 12 | Left_Ankle_Pitch | Knöchel Pitch |
| 13 | Left_Ankle_Roll | Knöchel Roll |
| **Rechtes Bein** |
| 14 | Right_Hip_Yaw | Hüfte Yaw |
| 15 | Right_Hip_Roll | Hüfte Roll |
| 16 | Right_Hip_Pitch | Hüfte Pitch |
| 17 | Right_Knee_Pitch | Knie Pitch |
| 18 | Right_Ankle_Pitch | Knöchel Pitch |
| 19 | Right_Ankle_Roll | Knöchel Roll |

---

## Vergleich: Original Student vs. Modified Student vs. Teacher

### Observation-Struktur

| Komponente | Teacher | Original Student | Modified Student |
|------------|---------|------------------|------------------|
| Proprio (65) | ✓ | ✓ | ✓ |
| Zukünftige Targets (1100) | ✓ (privileged) | ✗ | ✓ |
| Historie (10 Frames) | ✗ | ✓ | ✗ |
| Priv Info (78) | ✓ | ✗ | ✗ |
| **Total Input** | **1243** | **1023** | **1165** |

### Privilegierte Info (nur Teacher & Critic)

Der Teacher/Critic erhält zusätzlich `n_priv_info = 78`:

| Dimension | Beschreibung |
|-----------|--------------|
| 3 | Base Linear Velocity |
| 1 | Root Height |
| 27 | Key Body Positions (9 × 3) |
| 2 | Contact Mask (Fußkontakt) |
| 4 | Priv Latent Teil 1 |
| 1 | Priv Latent Teil 2 |
| 40 | Motor Strength + Action Delay (2 × 20) |

### Konzeptioneller Unterschied

| Aspekt | Original Student | Modified Student |
|--------|------------------|------------------|
| **Philosophie** | Zustandsschätzung | Zielplanung |
| **Zeitrichtung** | Vergangenheit → Gegenwart | Gegenwart → Zukunft |
| **Annahme** | Zustand kann aus Historie erschlossen werden | Zukünftige Ziele sind bekannt (Motion Capture) |
| **Vorteil** | Funktioniert ohne Motion-Targets | Bessere Antizipation der Bewegung |
| **Nachteil** | Reaktiv, nicht prädiktiv | Benötigt Motion-Target-System |

---

## Netzwerk-Architektur

```python
actor_hidden_dims = [512, 512, 256, 128]
critic_hidden_dims = [512, 512, 256, 128]
activation = 'silu'
layer_norm = True
motion_latent_dim = 128
```

### Actor Network (Modified Student)
```
Input (1165) → 512 → 512 → 256 → 128 → Output (20)
```

### Critic Network
```
Input (privileged) → 512 → 512 → 256 → 128 → Value (1)
```

---

## Konfigurationsparameter

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| `obs_type` | 'student_future' | Neuer Observationstyp mit Zukunfts-Targets |
| `history_len` | 0 | Keine Historie (Modified) |
| `tar_obs_steps` | [1,5,10,...,95] | 20 zukünftige Zeitpunkte |
| `action_scale` | 1 | Skalierung der Aktionen |
| `decimation` | 10 | Policy-Frequenz = Sim-Freq / decimation |
| `dt` | 0.002 | Simulationszeitschritt (500 Hz) |
| **Policy-Frequenz** | **50 Hz** | 500 / 10 = 50 Hz |

### Zeitliche Auflösung der Targets

Bei 50 Hz Policy-Frequenz entsprechen die `tar_obs_steps` folgenden Zeiten:

| Step | Frames | Zeit (ms) |
|------|--------|-----------|
| 1 | +1 | 20 ms |
| 5 | +5 | 100 ms |
| 10 | +10 | 200 ms |
| 20 | +20 | 400 ms |
| 50 | +50 | 1000 ms |
| 95 | +95 | 1900 ms |

Der Student sieht also Motion-Targets bis zu **~2 Sekunden in die Zukunft**.

---

## Referenz

Konfigurationsdatei: `legged_gym/legged_gym/envs/k1/k1_mimic_distill_config.py`

Klassen:
- `K1MimicStuRLCfg_modified` - Modified Student Environment Config
- `K1MimicStuRLCfgDAgger_modified` - Modified Student Training Config (DAgger)
