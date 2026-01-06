# Implementation Plan: Student Network mit Key Body Positions statt DOF Positions

**Erstellt am:** 18. November 2025  
**Ziel:** Student-Observation von DOF Positions (20) auf Key Body Positions (9×3 = 27) umstellen  
**Betrifft:** K1 Robot Student Training

---

## 📊 Änderungszusammenfassung

### Dimensionsänderung:
- **Aktuell:** `n_mimic_obs = 8 + 20 = 28` Dimensionen
- **Neu:** `n_mimic_obs = 8 + 27 = 35` Dimensionen (+7)
- **Mit History:** `28 × 11 = 308` → `35 × 11 = 385` (+77)

### Key Body Definitionen (aus config):
```python
key_bodies = [
    "left_hand_end_ball",    # Hand links
    "right_hand_end_ball",   # Hand rechts
    "left_foot_link",        # Fuß links
    "right_foot_link",       # Fuß rechts
    "right_outer_toe_link",  # Äußerer Zeh rechts
    "left_outer_toe_link",   # Äußerer Zeh links
    "right_inner_toe_link",  # Innerer Zeh rechts
    "left_inner_toe_link",   # Innerer Zeh links
    "Head_2"                 # Kopf
]
```

---

## 🎯 Vorteile dieser Änderung

1. ✅ **Räumliches Verständnis**: Netzwerk lernt Zielpositionen im 3D-Raum statt abstrakte Gelenkwinkel
2. ✅ **Kinematik-Unabhängigkeit**: Robuster gegen URDF-Unterschiede zwischen Sim und Real
3. ✅ **Mehr räumlicher Kontext**: 9 Körperpositionen geben bessere Information über die gewünschte Pose
4. ✅ **Konsistenz mit Teacher**: Teacher nutzt bereits Key Body Positions in privileged observations
5. ✅ **Implizite Inverse Kinematik**: Student lernt selbst, welche Gelenkwinkel zu den Zielpositionen führen

---

## 🔧 Benötigte Code-Änderungen

### 1. Config-Datei: `k1_mimic_distill_config.py`

#### Änderung A: K1MimicPrivCfg (Zeile ~27)
```python
# VORHER:
n_mimic_obs = 8 + const_num_actions # 20 for dof pos

# NACHHER:
n_mimic_obs = 8 + 3*9 # 9 key bodies * 3D positions (instead of 20 DOF)
```

#### Änderung B: K1MimicStuCfg (Zeile ~381)
```python
# VORHER:
n_mimic_obs = 8 + const_num_actions # 22 for dof pos

# NACHHER:
n_mimic_obs = 8 + 3*9 # 9 key bodies * 3D positions
```

#### Änderung C: K1MimicStuRLCfg (Zeile ~409)
```python
# VORHER:
n_mimic_obs = 8 + const_num_actions # 22 for dof pos

# NACHHER:
n_mimic_obs = 8 + 3*9 # 9 key bodies * 3D positions
```

**Neue Dimensionen:**
- `n_mimic_obs = 8 + 27 = 35`
- `n_obs_single = 35 + 65 = 100` (statt 93)
- `num_observations = 100 * 11 = 1100` (statt 1023)

---

### 2. Environment-Datei: `k1_mimic_distill.py`

#### Änderung in `_get_mimic_obs()` Methode (ab Zeile ~205)

**VORHER:**
```python
# v6, align mocap
mimic_obs_buf = torch.cat((
    root_pos[..., 2:3], # 1 dim
    roll, pitch, yaw, # 3 dims
    root_vel, # 3 dims
    root_ang_vel[..., 2:3], # 1 dim, yaw only
    dof_pos, # num_dof dims (20)
), dim=-1)[:, 0:1] # shape: (num_envs, 1, 8 + 20 = 28)


return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)
```

**NACHHER:**
```python
# v7, use key body positions instead of dof_pos for student observation
# Extract only the current timestep (index 0) of key body positions
key_body_pos_current = whole_key_body_pos[:, 0:1, :]  # shape: (num_envs, 1, 27)

mimic_obs_buf = torch.cat((
    root_pos[..., 2:3], # 1 dim
    roll, pitch, yaw, # 3 dims
    root_vel, # 3 dims
    root_ang_vel[..., 2:3], # 1 dim, yaw only
    key_body_pos_current, # 9 key bodies * 3D = 27 dims (instead of dof_pos)
), dim=-1)[:, 0:1] # shape: (num_envs, 1, 8 + 27 = 35)


return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)
```

**Wichtig:** Die Variable `whole_key_body_pos` ist bereits berechnet und enthält die Key Body Positions:
```python
whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
if self.global_obs:
    whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
```

---

## 📝 Detaillierte Observation-Struktur

### Student Observation (NEU):

```python
mimic_obs_buf (35 dims):
├─ root_pos[..., 2:3]           # 1  - Root height (Z)
├─ roll, pitch, yaw             # 3  - Root orientation
├─ root_vel                     # 3  - Root velocity (from motion)
├─ root_ang_vel[..., 2:3]       # 1  - Angular velocity (yaw only)
└─ key_body_pos_current         # 27 - 9 key body positions × 3D
   ├─ left_hand_end_ball        # 3  - (x, y, z)
   ├─ right_hand_end_ball       # 3  - (x, y, z)
   ├─ left_foot_link            # 3  - (x, y, z)
   ├─ right_foot_link           # 3  - (x, y, z)
   ├─ right_outer_toe_link      # 3  - (x, y, z)
   ├─ left_outer_toe_link       # 3  - (x, y, z)
   ├─ right_inner_toe_link      # 3  - (x, y, z)
   ├─ left_inner_toe_link       # 3  - (x, y, z)
   └─ Head_2                    # 3  - (x, y, z)

proprio_obs_buf (65 dims):
├─ base_ang_vel                 # 3  - Current angular velocity
├─ imu_obs (roll, pitch)        # 2  - IMU orientation
├─ dof_pos (reindexed)          # 20 - Current joint positions
├─ dof_vel (reindexed)          # 20 - Current joint velocities
└─ last_action                  # 20 - Previous action

Total per timestep: 35 + 65 = 100 dims
With history (10 steps): 100 × 11 = 1100 dims
```

---

## ⚙️ Koordinatensystem-Wichtigkeit

**Aktuelles Verhalten** (abhängig von `self.global_obs`):
- `global_obs = False` (Standard): Key Body Positions sind **relativ zum Root** (lokal)
- `global_obs = True`: Key Body Positions sind **global**

**Empfehlung:** Bei `global_obs = False` bleiben (aktueller Standard), da:
- Konsistenter mit propriozeptiven Daten
- Invariant gegenüber globaler Position
- Einfacher zu lernen für das Netzwerk

---

## 🧪 Testing & Validation

### Nach Implementation testen:

1. **Dimensionen prüfen:**
   ```python
   # In k1_mimic_distill.py sollte ausgegeben werden:
   print(f"[K1 Obs Debug] mimic_obs shape: {mimic_obs.shape}")
   # Erwarteter Output: torch.Size([num_envs, 35])
   ```

2. **Observation Buffer Größe:**
   ```python
   print(f"obs_buf shape: {self.obs_buf.shape}")
   # Erwarteter Output: torch.Size([num_envs, 1100])
   ```

3. **Vergleich: Vor/Nach Training:**
   - Baseline: Mit DOF Positions trainieren (aktuelles System)
   - Experiment: Mit Key Body Positions trainieren (neue Variante)
   - Metriken vergleichen: Tracking-Genauigkeit, Stabilität, Konvergenzgeschwindigkeit

---

## 🚀 Implementierungs-Reihenfolge

1. ✅ **Backup erstellen** der aktuellen Config und Environment
2. ✅ **Config ändern:** Alle drei Klassen (PrivCfg, StuCfg, StuRLCfg)
3. ✅ **Code ändern:** `_get_mimic_obs()` Methode
4. ✅ **Dimensionen testen:** Kurzer Test-Run ohne Training
5. ✅ **Baseline-Training:** Aktuelles System als Referenz trainieren
6. ✅ **Experiment-Training:** Neues System mit Key Body Positions trainieren
7. ✅ **Vergleich:** Beide Modelle evaluieren und vergleichen

---

## 📌 Zusätzliche Notizen

### Warum nicht auch für Teacher?
- Teacher hat bereits Zugriff auf Key Body Positions in `priv_info`
- Teacher bekommt 20 future timesteps mit Key Body Positions in `priv_mimic_obs`
- Änderung betrifft nur Student-Observation

### Potenzielle Probleme:
1. **Noise:** Key Body Positions könnten in Simulation rauschen
   - Lösung: Ggf. smoothing oder filtering hinzufügen
2. **Koordinatensystem:** Muss konsistent sein zwischen Sim und Real
   - Lösung: Immer relativ zum Root verwenden (`global_obs = False`)
3. **Dimension Mismatch:** Beim Laden alter Checkpoints
   - Lösung: Neue Experimente mit neuem `exptid` starten

---

## 📚 Referenzen im Code

**Relevante Dateien:**
- `/home/nao/Documents/TWIST/legged_gym/legged_gym/envs/k1/k1_mimic_distill_config.py` (Zeilen 27, 381, 409)
- `/home/nao/Documents/TWIST/legged_gym/legged_gym/envs/k1/k1_mimic_distill.py` (Zeilen 200-215)
- `/home/nao/Documents/TWIST/legged_gym/legged_gym/envs/base/humanoid_mimic.py` (Basis-Klasse)

**Key Body IDs werden definiert in:**
- `_key_body_ids_motion`: Indices für Motion Library
- `_key_body_ids`: Indices für Simulation Bodies

---

## ✅ Checkliste vor Implementation

- [ ] Aktuelles System komplett trainieren (Baseline)
- [ ] Baseline-Checkpoints sichern
- [ ] Git Branch erstellen: `feature/key-body-positions-student`
- [ ] Config-Änderungen vornehmen
- [ ] Code-Änderungen vornehmen
- [ ] Dimension-Test durchführen (kurzer Run)
- [ ] Volles Training starten
- [ ] Evaluation und Vergleich
- [ ] Bei Erfolg: in main Branch mergen

---

**Status:** ⏸️ GEPLANT - Warte auf Baseline-Referenz-Training  
**Priorität:** 🟢 HOCH - Vielversprechende Verbesserung  
**Geschätzter Aufwand:** ~30 Minuten Implementation + Training Zeit
