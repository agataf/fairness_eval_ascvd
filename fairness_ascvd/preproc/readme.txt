VARIABLES EXTRACTED FROM COHORTS

general/demographic:

* cohort_pid
* racegrp
* gender
* age
* study

measured at exam 1:

* cursmoke: currently a smoker
* hyptmdsr: taking blood pressure-lowering medication
* cholmed: taking cholesterol-lowering medication (e.g. statin)
* diabt126: diabetes
* totchol: total cholesterol level
* ldlc: LDL cholesterol level
* trigly: triglyceride level
* hdlc: HDL cholesterol level
* sysbp: systolic blood pressure
* diabp: diastolic blood pressure
* bmi: body mass index

conditions before or at exam 1 (binary variables):

* prevproc: "Prevalent Cor Revas procedure" (coronary bypass surgery?)
* prevchf: "Prevalent Congestive Heart Failure"
* prevap: "Prevalent Angina (Study)"
* prevmi: "Prevalent MI by history or ECG" 
* prevchd: "Prevalent CHD (MI, SMI, Procedures)" 
* prevstrk: "Prevalent Stroke" 
* prevafib: Prevalent atrial fibrillation

* prevang: angioplasty (goes into prevproc!!)


exclusion criteria : myocardial infarction, stroke, coronary bypass surgery or angioplasty, congestive heart failure or atrial fibrillation

events after exam 1:

* mi: non-lethal myocardial infarction
* strk: non-lethal stroke
* chddeath: death from CHD
* strkdeath: death from stroke
* death: death from any cause

time to events after exam 1 (in years, counted from day of exam 1):

* timetomi
* timetostrk
* timetochddeath
* timetostrkdeath
* timetodth
* lastexam: last record of in-person examination
* lastfu: last follow-up (typically phone interview)

INPUT VARIABLES TO MODELS:

* cohort_pid
* study
* gender_male
* age
* race_black
* grp

* hdlc
* ldlc
* trigly
* totchol
* cursmoke
* diabt126
* unrxsbp
* rxsbp
* ascvd_10yr: ASCVD event (mi OR strk OR chddeath OR strkdeath) within 10 years of exam 1
* censored_10yr: censored before 10 years of exam 1 (last follow-up, last exam, or non-ASCVD death before 10 years)
* event_time_10yr: time to ASCVD event within 10 years of exam 1

