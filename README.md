# student-placement-prediction
### TASK 1: Data Cleaning & Preprocessing (With Numbers)
- Dataset: 300 rows, 12 columns
- Missing Found: cgpa=5 (1.6%), internship=12 (4%), certification=18 (6%), Duplicates=7
- Code: df['cgpa'].fillna(df['cgpa'].median(), inplace=True) # median=7.8
- After Cleaning: Missing=0, Final rows=293, Verified via df.isnull().sum()

### TASK 2: Feature Engineering (With Example)
- New Feature: academic_score = (cgpa*10 + aptitude_score)/2
- Worked Example: Student with cgpa 8.2, aptitude 85 => (82+85)/2 = 83.5
- Encoding: branch -> One-Hot Encoding (branch_CSE, branch_ECE, branch_MECH)
- Scaling: StandardScaler on cgpa, aptitude_score
- Result: Accuracy Before 68%, After 84.2% (+16.2% improvement)

### TASK 3: Hypothesis, Assumptions & Risk (With Evidence)
- H0: Internship has no relation. H1: Internship increases placement.
- Evidence: Internship=Yes 145 students -> 120 placed (82.7%), No=148 -> 75 placed (50.6%). p-value=0.001 <0.05 => H1 proved
- Assumption: CGPA matters. Evidence: Placed avg CGPA 8.1 vs Not Placed 6.9 (Diff 1.2), Corr=0.72
- Risk Mitigation: Overfitting-> max_depth=8 + K-Fold k=5, Imbalance-> SMOTE (200 vs 93), Leakage-> Removed salary_offered column
