"""ETL – WRI Electric School Bus dataset v10 -> clean tables + KPI values.
Usage: python3 -I etl.py <input.xlsx> <output_dir>
"""
import sys, json, re
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
log = {}

# ---------- EXTRACT ----------
sheets = pd.read_excel(src, sheet_name=None)
d = sheets['1. District-level data'].copy()
b = sheets['2. Bus-level data'].copy()
s = sheets['3. State-level data'].copy()
ut = sheets['4. Utilities'].copy()
co = sheets['5. Counties'].copy()
for df in (d, b, s, ut, co):
    df.columns = [c.strip() for c in df.columns]
log['extract_rows'] = {'district': len(d), 'bus': len(b), 'state': len(s), 'utilities': len(ut), 'counties': len(co)}

# ---------- TRANSFORM: districts ----------
n0 = len(d)
d = d[d['1a. State'].notna()].copy()                       # drop the grand-total row (no state)
log['district_total_rows_dropped'] = n0 - len(d)
d['state'] = d['1a. State'].str.strip().str.upper()         # 'New York' -> 'NEW YORK'
d['has_esb'] = d['0a. Has Committed ESBs?'].str.strip().str.lower().eq('yes')
d['has_lea_id'] = d['1j. Has LEA ID?'].str.strip().str.lower().eq('yes')
log['text_case_fixed'] = int((d['1a. State'] != d['state']).sum())
for c in ['5c. Quartile: percent non-white and/or Hispanic', '5e. Quartile: percent low-income',
          '5g. Quartile: PM2.5 concentration', '5i. Quartile: ozone concentration', '5m. Quartile: average rate of asthma']:
    bad = d[c] > 4                                          # stray count (3294/3295) in a 1-4 field
    log.setdefault('quartile_outliers_set_to_null', 0)
    log['quartile_outliers_set_to_null'] += int(bad.sum())
    d.loc[bad, c] = np.nan
d['locale'] = d['1p. Locale broad type (name)'].fillna('Unknown')
d['pod'] = d['5q. WRI Priority Outreach District (POD)?'].map({True: 'Yes', False: 'No'}).fillna('Unknown')
num = {'students': '4b. Number of students in district', 'buses_total': '2a. Total number of buses',
       'esb_committed': '3a. Number of ESBs committed', 'esb_operating': '3f. Number of ESBs operating',
       'esb_delivered': '3e. Number of ESBs delivered', 'esb_ordered': '3d. Number of ESBs ordered',
       'esb_awarded': '3c. Number of ESBs awarded', 'pm25': '5f. PM2.5 concentration',
       'low_income_pct': '5d. Percent low-income (200% of federal poverty level)',
       'median_income': '4f. Median household income', 'lat': '1s. Latitude', 'lon': '1t. Longitude'}
for k, c in num.items():
    d[k] = pd.to_numeric(d[c], errors='coerce')
for k in ['esb_committed', 'esb_operating', 'esb_delivered', 'esb_ordered', 'esb_awarded']:
    d[k] = d[k].fillna(0).astype(int)                       # missing = no ESB
log['students_missing_kept_null'] = int(d['students'].isna().sum())
log['buses_total_missing_kept_null'] = int(d['buses_total'].isna().sum())
d.loc[d['buses_total'] == 0, 'buses_total'] = np.nan        # 0 buses = unusable denominator
d['esb_share'] = (d['esb_committed'] / d['buses_total']).where(d['buses_total'].notna())
d['q_lowinc'] = d['5e. Quartile: percent low-income']
d['q_pm25'] = d['5g. Quartile: PM2.5 concentration']
d['high_need'] = (d['q_lowinc'] == 4) & (d['q_pm25'] == 4)
d['interest'] = d['6a. Has any expression of interest in ESBs?'].fillna(0).astype(float)
d['applied_not_awarded'] = d['6e. Applied for ESB funding but not awarded'].notna()
d['sust_commit'] = d['6g. School district sustainability commitment'].notna()
d['climate_mayors'] = d['6f. Member of Climate Mayors Electric Vehicle Purchasing Collaborative?'].astype(str).str.lower().isin(['yes', 'true', '1', '1.0'])
d['lea_id'] = d['1c. LEA ID']
d['lea_name'] = d['1b. Local Education Agency (LEA) or entity name']

# ---------- TRANSFORM: buses ----------
b['state'] = b['1a. State'].str.strip().str.upper()
b['lea_id'] = b['1c. LEA ID']
b['status'] = b['3n. Current status of bus'].str.strip()
qa = b['3p. Quarter awarded'].astype(str).str.extract(r'(\d{4})\s*Q?(\d)?')
b['year_awarded'] = pd.to_numeric(qa[0], errors='coerce')
b['quarter_awarded'] = pd.to_numeric(qa[1], errors='coerce')
b['oem'] = b['3t. Bus OEM'].fillna('Unknown').str.strip()
b['bus_type'] = b['3w. Type'].fillna('Unknown').astype(str).str.strip()
b['funding1'] = b['3z. Funding source 1'].fillna('Unknown').str.strip()
def group(f):
    if re.search(r'Clean School Bus|Clean Heavy-Duty', f): return 'EPA (federal)'
    if 'VW' in f: return 'VW Settlement'
    if f == 'Unknown': return 'Unknown'
    return 'State / other'
b['funding_group'] = b['funding1'].map(group)
b['amount_bus'] = pd.to_numeric(b['3ab. Dollar amount toward bus'], errors='coerce')
b['stalled'] = (b['status'] == 'Awarded') & (b['year_awarded'] <= 2023)   # awarded >= 2 years before data cut-off
log['bus_missing_year'] = int(b['year_awarded'].isna().sum())
log['bus_orphans_without_district'] = int((~b['lea_id'].isin(d['lea_id'])).sum())
log['bus_identical_rows_(same order, different buses)'] = int(b.duplicated().sum())
chk = b.groupby('lea_id').size().reindex(d['lea_id']).fillna(0).values
log['reconciliation_bus_rows_vs_district_committed_mismatches'] = int((chk != d['esb_committed'].values).sum())
log['reconciliation_total_bus_rows'] = int(len(b)); log['reconciliation_total_district_committed'] = int(d['esb_committed'].sum())
b['bus_id'] = np.arange(1, len(b) + 1)

# ---------- TRANSFORM: states ----------
s = s[s['1a. State'].notna() & ~s['1a. State'].str.startswith('#') & ~s['1a. State'].str.contains('No state')].copy()
s['state'] = s['1a. State'].str.upper()
s['buses_total'] = s['2a. Total number of school buses (WRI 2024)'].replace(0, np.nan)
s['esb_committed'] = s['3a. Number of committed ESBs']
s['avg_price_type_c'] = s['3j. Average purchase price of Type C ESB (2022-2025)']
s['students_riding_esb'] = s['8a. Approximate total number of students riding ESBs']
s['esb_share'] = s['esb_committed'] / s['buses_total']
s['mandate'] = s['8d. Pupil transportation requirement?']
log['states_rows'] = len(s)

# ---------- LOAD ----------
dim_state = s[['state', 'buses_total', 'esb_committed', 'esb_share', 'avg_price_type_c', 'students_riding_esb']]
dim_district = d[['lea_id', 'lea_name', 'state', 'locale', 'lat', 'lon', 'students', 'buses_total', 'esb_committed',
                  'esb_awarded', 'esb_ordered', 'esb_delivered', 'esb_operating', 'esb_share', 'low_income_pct',
                  'q_lowinc', 'pm25', 'q_pm25', 'median_income', 'pod', 'high_need', 'has_esb', 'interest',
                  'applied_not_awarded', 'sust_commit']]
fact_bus = b[['bus_id', 'lea_id', 'state', 'status', 'year_awarded', 'quarter_awarded', 'oem', 'bus_type',
              'funding1', 'funding_group', 'amount_bus', 'stalled']]
dim_state.to_csv(f'{out}/dim_state.csv', index=False)
dim_district.to_csv(f'{out}/dim_district.csv', index=False)
fact_bus.to_csv(f'{out}/fact_bus.csv', index=False)

# ---------- KPIs ----------
K = {}
K['committed'] = int(len(b))
K['k1_adoption_rate'] = float(s['esb_committed'].sum() / s['buses_total'].sum())
hn = d[d['high_need']]
K['k2_high_need_districts'] = int(len(hn)); K['k2_high_need_with_esb'] = int(hn['has_esb'].sum())
K['k2_priority_coverage'] = float(hn['has_esb'].mean())
K['k2_students_uncovered'] = float(hn.loc[~hn['has_esb'], 'students'].sum())
K['k3_epa_dependence'] = float((b['funding_group'] == 'EPA (federal)').mean())
K['k4_stalled_rate'] = float(b['stalled'].mean()); K['k4_stalled_n'] = int(b['stalled'].sum())
K['k5_operating_rate'] = float((b['status'] == 'Operating').mean())
K['k5_delivered_or_operating'] = float(b['status'].isin(['Operating', 'Delivered']).mean())
K['districts'] = int(len(d)); K['districts_with_esb'] = int(d['has_esb'].sum())
log['kpis'] = K

# ---------- Priority score (decision model, Lab 3 'Choice') ----------
cand = d[~d['has_esb'] & d['q_lowinc'].notna() & d['students'].gt(0)].copy()
r = lambda x: x.rank(pct=True)
cand['ready'] = (cand['interest'] + cand['applied_not_awarded'].astype(float) + cand['sust_commit'].astype(float)) / 3
cand['score'] = 100 * (0.30 * r(cand['low_income_pct'].fillna(cand['low_income_pct'].median())) +
                       0.25 * r(cand['pm25'].fillna(cand['pm25'].median())) +
                       0.25 * r(cand['students']) + 0.20 * cand['ready'])
top = cand.sort_values('score', ascending=False).head(50)
top[['lea_id', 'lea_name', 'state', 'locale', 'students', 'low_income_pct', 'pm25', 'ready', 'score']].to_csv(f'{out}/top50_priority_districts.csv', index=False)
log['priority_candidates'] = int(len(cand))

# ---------- New York focus (project problem: school-bus exhaust exposure and Clean School Zones) ----------
WHO_PM25 = 5.0                                              # WHO 2021 annual guideline, ug/m3
ny = d[d['state'] == 'NEW YORK'].copy()
nyb = b[b['state'] == 'NEW YORK'].copy()
ny['exposure'] = ny['pm25'] * ny['students']                # simple exposure score: PM2.5 x students
thr = ny['pm25'].quantile(0.75)                             # priority = top quartile of NY PM2.5
ny['priority_zone'] = ny['pm25'] >= thr
ny_pm = ny[ny['pm25'].notna() & ny['students'].gt(0)]
NY = {}
NY['districts'] = int(len(ny)); NY['districts_with_pm25'] = int(ny['pm25'].notna().sum())
NY['fleet_wri_2024'] = float(s.loc[s['state'] == 'NEW YORK', 'buses_total'].iloc[0])
NY['k1_mean_pm25_student_weighted'] = float((ny_pm['pm25'] * ny_pm['students']).sum() / ny_pm['students'].sum())
NY['k1_ratio_to_who'] = NY['k1_mean_pm25_student_weighted'] / WHO_PM25
NY['k1_share_districts_above_who'] = float((ny['pm25'].dropna() > WHO_PM25).mean())
NY['k2_committed'] = int(len(nyb)); NY['k2_committed_share'] = NY['k2_committed'] / NY['fleet_wri_2024']
NY['k2_operating'] = int((nyb['status'] == 'Operating').sum()); NY['k2_operating_share'] = NY['k2_operating'] / NY['fleet_wri_2024']
NY['k3_threshold_pm25'] = float(thr); NY['k3_priority_districts'] = int(ny['priority_zone'].sum())
NY['k3_priority_students'] = float(ny.loc[ny['priority_zone'], 'students'].sum())
NY['k3_priority_share_students'] = NY['k3_priority_students'] / float(ny['students'].sum())
NY['k4_zone_baseline'] = 0                                  # no clean school zone recorded yet: tracked by the association
NY['k5_operating_rate'] = NY['k2_operating'] / NY['k2_committed']
log['ny'] = NY
ny[['lea_id', 'lea_name', 'locale', 'lat', 'lon', 'students', 'buses_total', 'esb_committed', 'esb_operating', 'pm25', 'q_pm25',
    'low_income_pct', 'exposure', 'priority_zone']].to_csv(f'{out}/ny_districts.csv', index=False)
nyb.to_csv(f'{out}/ny_buses.csv', index=False)
json.dump(log, open(f'{out}/etl_log.json', 'w'), indent=1, default=float)
print(json.dumps(log, indent=1, default=float))
