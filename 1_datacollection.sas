/**********************************************************************************************
*  TO BE SET BY USER
**********************************************************************************************/

* Specify library location;
 libname tds ''; /* local location for data download */ 
 libname iclink ''; /* location of iclink table, which is placed in data/iclink.sas7bdat to start */

* Login to WRDS;
%let wrds = wrds-cloud.wharton.upenn.edu 4016;
options comamid = TCP remote = WRDS;
signon username = /* YOUR WRDS USERNAME */ password = /* YOUR WRDS PASSWORD */;

* Load macros;
* NEED TO CHANGE THIS TO YOUR LOCAL LOCATION. THEY ARE CURRENTLY IN "macros" FOLDER;
rsubmit; 
%include 'winsorize.sas';
%include 'outliers_iqr.sas';
%include 'get_EAs.sas';
%include 'createlags.sas';
%include 'get_fundq.sas';
%include 'get_crspmsf_delist.sas';
endrsubmit;


/**********************************************************************************************
*  Get IBES data
**********************************************************************************************/

rsubmit;
* List of IBES EPS announcement dates;
proc sql;
	create table ibes1 as select TICKER, PENDS, ANNDATS as ANNDATS_ACTUAL, VALUE as EPS, PDICITY
	from ibes.act_epsus
	where MEASURE = 'EPS' and USFIRM = 1 and CURR_ACT = 'USD' and
		missing(EPS) = 0 and
		(1989 <= year(PENDS))
	;
quit;

* Check no duplicates on this key;
proc sort data = ibes1;
	by TICKER PENDS ANNDATS_ACTUAL PDICITY;
run;

* Break into quarterly and annual announcement dates;
data ibes1q;
	set ibes1;
	if PDICITY = 'QTR';
	drop PDICITY;
run;
data ibes1a;
	set ibes1;
	if PDICITY = 'ANN';
	drop PDICITY;
run;

* For each EPS announcement-frequency, calculate the period-end and reported EPS at the next announcements;
* Note: for quarterly announcements, we only take firm-years that are FY-ends;
proc sort data = ibes1q;
	by TICKER descending PENDS;
run;
proc sort data = ibes1a;
	by TICKER descending PENDS;
run;
**** QUARTERLY;
data ibes1_q1;
	set ibes1q;
	by TICKER;
		LEAD_PENDS = ifn(first.TICKER, ., lag(PENDS)); 
		LEAD_EPS = ifn(first.TICKER, ., lag(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(first.TICKER, ., lag(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
proc sql;
	create table ibes1_q1 as select a.*
	from ibes1_q1 as a inner join ibes1a as b
	on a.TICKER = b.TICKER and a.PENDS = b.PENDS
	;
quit;
data ibes1_q2;
	set ibes1q;
	by TICKER;
		LEAD_PENDS = ifn(lag2(TICKER) ^= TICKER, ., lag2(PENDS)); 
		LEAD_EPS = ifn(lag2(TICKER) ^= TICKER, ., lag2(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag2(TICKER) ^= TICKER, ., lag2(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
proc sql;
	create table ibes1_q2 as select a.*
	from ibes1_q2 as a inner join ibes1a as b
	on a.TICKER = b.TICKER and a.PENDS = b.PENDS
	;
quit;
data ibes1_q3;
	set ibes1q;
	by TICKER;
		LEAD_PENDS = ifn(lag3(TICKER) ^= TICKER, ., lag3(PENDS)); 
		LEAD_EPS = ifn(lag3(TICKER) ^= TICKER, ., lag3(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag3(TICKER) ^= TICKER, ., lag3(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
proc sql;
	create table ibes1_q3 as select a.*
	from ibes1_q3 as a inner join ibes1a as b
	on a.TICKER = b.TICKER and a.PENDS = b.PENDS
	;
quit;
data ibes1_q4;
	set ibes1q;
	by TICKER;
		LEAD_PENDS = ifn(lag4(TICKER) ^= TICKER, ., lag4(PENDS)); 
		LEAD_EPS = ifn(lag4(TICKER) ^= TICKER, ., lag4(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag4(TICKER) ^= TICKER, ., lag4(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
proc sql;
	create table ibes1_q4 as select a.*
	from ibes1_q4 as a inner join ibes1a as b
	on a.TICKER = b.TICKER and a.PENDS = b.PENDS
	;
quit;
**** ANNUAL;
data ibes1_a1;
	set ibes1a;
	by TICKER;
		LEAD_PENDS = ifn(first.TICKER, ., lag(PENDS)); 
		LEAD_EPS = ifn(first.TICKER, ., lag(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(first.TICKER, ., lag(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
data ibes1_a2;
	set ibes1a;
	by TICKER;
		LEAD_PENDS = ifn(lag2(TICKER) ^= TICKER, ., lag2(PENDS)); 
		LEAD_EPS = ifn(lag2(TICKER) ^= TICKER, ., lag2(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag2(TICKER) ^= TICKER, ., lag2(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
data ibes1_a3;
	set ibes1a;
	by TICKER;
		LEAD_PENDS = ifn(lag3(TICKER) ^= TICKER, ., lag3(PENDS)); 
		LEAD_EPS = ifn(lag3(TICKER) ^= TICKER, ., lag3(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag3(TICKER) ^= TICKER, ., lag3(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
data ibes1_a4;
	set ibes1a;
	by TICKER;
		LEAD_PENDS = ifn(lag4(TICKER) ^= TICKER, ., lag4(PENDS)); 
		LEAD_EPS = ifn(lag4(TICKER) ^= TICKER, ., lag4(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag4(TICKER) ^= TICKER, ., lag4(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;
data ibes1_a5;
	set ibes1a;
	by TICKER;
		LEAD_PENDS = ifn(lag5(TICKER) ^= TICKER, ., lag5(PENDS)); 
		LEAD_EPS = ifn(lag5(TICKER) ^= TICKER, ., lag5(EPS));
		LEAD_ANNDATS_ACTUAL = ifn(lag5(TICKER) ^= TICKER, ., lag5(ANNDATS_ACTUAL));
	if missing(LEAD_PENDS) or missing(LEAD_EPS) then delete; /* delete observations we cant use */
	format LEAD_PENDS LEAD_ANNDATS_ACTUAL YYMMDDN8.;
run;

* Collect analyst forecast data that satisfy criteria for merge;
data det_epsus_a;
	set ibes.det_epsus;
	where FPI in ('1', '2', '3', '4', '5') and /* annual forecasts only*/
		CURR_ACT = 'USD' and /* delete duplicates from multiple currencies */
		not missing(VALUE);
run;
data det_epsus_q;
	set ibes.det_epsus;
	where FPI in ('6', '7', '8', '9') and /* quarterly forecasts only*/
		CURR_ACT = 'USD' and /* delete duplicates from multiple currencies */
		not missing(VALUE);
run;

* Make datasets of lags of EPS for each frequency;
data epslags_q;
	set ibes1q;
	keep TICKER PENDS EPS;
run;
data epslags_a;
	set ibes1a;
	keep TICKER PENDS EPS;
run;
%createlags(data = epslags_q, idvar = TICKER, datevar = PENDS, vars = EPS, nlags = 2);
%createlags(data = epslags_a, idvar = TICKER, datevar = PENDS, vars = EPS, nlags = 2);

endrsubmit;	



/**********************************************************************************************
*  Prepare COMPUSTAT data
**********************************************************************************************/

rsubmit;
* Data extraction step; 
data compustat_a;
	set compa.funda (keep = GVKEY DATADATE INDFMT POPSRC CONSOL DATAFMT AT);
	where DATADATE > '01JAN1985'd and DATAFMT = 'STD' and POPSRC = 'D' and INDFMT = 'INDL' 
		and CONSOL ='C' and AT > 0;
	T_DATADATE = year(DATADATE);
	QTR = qtr(DATADATE);
	LOG_AT = log(AT);
	drop DATAFMT POPSRC INDFMT CONSOL AT;
run;
%get_fundq(outdata = compustat_q, vars = ATQ, startyr = 1985, endyr = 2022);
data compustat_q;
	set compustat_q;
	where ATQ > 0;
	T_DATADATE = year(DATADATE);
	QTR = qtr(DATADATE);
	LOG_AT = log(ATQ);	
	drop ATQ PERMNO;
run;

* Create lags of variables;
%createlags(data = compustat_a, idvar = GVKEY, datevar = DATADATE, vars = LOG_AT, nlags = 2);
%createlags(data = compustat_q, idvar = GVKEY, datevar = DATADATE, vars = LOG_AT, nlags = 2);
endrsubmit;



/**********************************************************************************************
*  Prepare CRSP dataset
**********************************************************************************************/

rsubmit;
%get_crspmsf_delist(outdata=crspmsf_delist);
endrsubmit;



/**********************************************************************************************
*  Prepare WRDS Ratio datasets
**********************************************************************************************/

* Variables of interest;
rsubmit;
%let ratiovars = CAPEI BE bm evm pe_exi pe_inc ps pcf dpr 
		npm opmbd opmad gpm ptpm cfm roa roe roce aftret_eq aftret_invcapx 
		aftret_equity pretret_noa pretret_earnat GProf equity_invcap debt_invcap 
		totdebt_invcap capital_ratio int_totdebt cash_lt invt_act rect_act debt_at 
		debt_ebitda short_debt curr_debt lt_debt profit_lct ocf_lct cash_debt fcf_ocf lt_ppent 
		dltt_be debt_assets debt_capital de_ratio intcov intcov_ratio cash_ratio quick_ratio 
		curr_ratio cash_conversion inv_turn at_turn rect_turn pay_turn sale_invcap sale_equity 
		rd_sale adv_sale accrual ptb divyield
	;
endrsubmit;

* Create quarterly and annual datasets of WRDS ratios dropping variables of no interest;
rsubmit;
data firm_ratio_q;
	set wrdsapps.firm_ratio;
	* Replace missing dividends to zero;
	dpr = ifn(missing(dpr), 0, dpr);
	divyield = ifn(missing(divyield), 0, divyield);
	* Replace missing interest with zero;
	int_totdebt = ifn(missing(int_totdebt), 0, int_totdebt);
	keep gvkey adate qdate public_date &ratiovars;
run;
proc sort data = firm_ratio_q;
	by GVKEY ADATE QDATE PUBLIC_DATE;
run;
proc sort data = firm_ratio_q(drop = PUBLIC_DATE) nodupkey;
	by GVKEY ADATE QDATE; /* keep only earliest observation montly within in quarter */
run;
data firm_ratio_a;
	set firm_ratio_q;
	where ADATE = QDATE; /* FY-end only */
	drop ADATE;
run;
data firm_ratio_q;
	set firm_ratio_q;
	drop ADATE;
run;

* Create lags of each variable;
%createlags(data = firm_ratio_q, idvar = gvkey, datevar = qdate, vars = &ratiovars, nlags = 2);
%createlags(data = firm_ratio_a, idvar = gvkey, datevar = qdate, vars = &ratiovars, nlags = 2);
endrsubmit;



/**********************************************************************************************
*  Macro to merge each horizon dataset with everything
**********************************************************************************************/

%macro form_df(freq=, h=);

%syslput _all_ ;
rsubmit;

* Collect analyst forecasts for the next fiscal year end that occur within 45 days of previous
announcement. (Need SELECT DISTINCT b/c duplicate obs due to IBES reviews);
proc sql;
	create table ibes2 as select distinct a.*, b.ANALYS, b.PDF, b.FPI, b.VALUE as FCAST, 
		b.ANNDATS, b.ANNTIMS, b.CURR_ACT, b.ACTDATS, b.ACTTIMS
	from ibes1_&freq&h as a inner join det_epsus_&freq as b
	on a.TICKER = b.TICKER and 
		a.LEAD_PENDS = b.FPEDATS and
		(1 <= intck('day', a.ANNDATS_ACTUAL, b.ANNDATS) <= 45) and
		(intck('day', b.ANNDATS, a.LEAD_ANNDATS_ACTUAL) > 0) and /* ensure forecast isn't after next announcement */
		a.LEAD_EPS = b.ACTUAL /* this should be true for almost all obs */
	order by TICKER, LEAD_PENDS, ANALYS, ANNDATS, ANNTIMS, ACTDATS desc, ACTTIMS desc
	;
quit;

* Drop all related observations to duplicates that occur early in sample for strange reason;
proc sort data = ibes2 uniqueout = ibes3 nouniquekey; 
	by TICKER LEAD_PENDS ANALYS ANNDATS ANNTIMS ACTDATS ACTTIMS; 
run; 

* Delete duplicates due to multiple observations on the same announcement time with
different activation dates;
proc sort data = ibes3 out = ibes4(drop = ACTTIMS ACTDATS ANNTIMS CURR_ACT) nodupkey;
	by TICKER LEAD_PENDS ANALYS ANNDATS ANNTIMS;
run;

* If analyst issues multiple forecasts, keep earliest one;
proc sort data = ibes4 nodupkey;
	by TICKER LEAD_PENDS ANALYS;
run;

* Count number of firm-year observations;
proc sort data = ibes4 out = junk nodupkey;
	by TICKER LEAD_PENDS;
run;

* Merge with ICLINK table to get PERMNOs, keeping only the best scores (see get_iclink.sas
for detail on scores);
proc upload data = iclink.iclink out = iclink;
run;
proc sql;
	create table ibes5 as select a.*, b.PERMNO, b.SCORE as IBES_SCORE
	from ibes4 as a inner join iclink as b
	on a.TICKER = b.TICKER
	where (0 <= b.SCORE <= 5)
	order by TICKER, LEAD_PENDS, ANALYS, IBES_SCORE
	;
quit;
proc sort data = ibes5 out = ibes5(drop = IBES_SCORE) nodupkey;
	by TICKER LEAD_PENDS ANALYS;
run;

* Delete small number observations that have different IBES tickers 
associated to the same PERMNO (not sure why this occurs);
proc sort data = ibes5 uniqueout = ibes6 nouniquekey;
	by PERMNO LEAD_PENDS ANALYS;
run;

* Get GVKEYs from CRSP-COMPUSTAT linking table;
proc sql;
	create table ibes7 as select a.*, b.UGVKEY
	from ibes6 as a inner join crspa.ccmxpf_lnkused as b
	on a.PERMNO = b.UPERMNO and
		b.ULINKTYPE in ('LC', 'LU') and
 		b.USEDFLAG = 1 and 
		(b.ULINKDT <= a.PENDS or b.ULINKDT = .B) and
		(b.ULINKENDDT >= a.LEAD_PENDS or b.ULINKENDDT = .E) and
		b.ULINKPRIM in ('P', 'C')
	order by TICKER, LEAD_PENDS, ANALYS
	;
quit;

* Should be no duplicates in either of these;
proc sort data = ibes7 nodupkey;
	by PERMNO LEAD_PENDS ANALYS;
run;
proc sort data = ibes7 nodupkey;
	by UGVKEY LEAD_PENDS ANALYS;
run;

* Get rid of tiny number of observations that have multiple LEAD_EPSs for one GVKEY-PEAD,
which must have to do with some kind of ticker change or IBES mistake;
proc sort data = ibes7 out = ibes8 nodupkey;
	by UGVKEY PENDS LEAD_EPS;
run;
proc sort data = ibes8 nouniquekey uniqueout = ibes9;
	by UGVKEY PENDS;
run;
proc sql;
	create table ibes10 as select b.*
	from ibes9 as a inner join ibes7 as b
	on a.UGVKEY = b.UGVKEY and a.PENDS = b.PENDS
	order by UGVKEY, LEAD_PENDS, ANALYS
	;
quit;



/**********************************************************************************************
* MERGE COMPUSTAT DATA WITH IBES
**********************************************************************************************/

* Merge COMPUSTAT and IBES;
proc sql;
	create table ibescomp1 as select *
	from ibes10 as a inner join compustat_&freq as b
	on a.UGVKEY = b.GVKEY and a.PENDS = b.DATADATE 
	order by UGVKEY, PENDS, ANALYS
	;
quit;

* Check for no duplicates;
proc sort data = ibescomp1 out = ibescomp2(drop = UGVKEY) nodupkey;
	by GVKEY PENDS ANALYS;
run; 



/**********************************************************************************************
* MERGE WITH CRSP DATA
**********************************************************************************************/

* Delete PERMNOs that arent ordinary shares and get SIC industry code;
proc sql;
	create table ibescrspcomp1 as select a.*, b.HSHRCD, 
				(substr(cat(b.HSICCD),1,1)) as SIC_1,
				(substr(cat(b.HSICCD),1,2)) as SIC_2
	from ibescomp2 as a inner join crsp.msfhdr as b
	on a.PERMNO = b.PERMNO and (b.BEGPRC <= a.ANNDATS_ACTUAL <= b.ENDPRC)
	having HSHRCD in (10,11)
	;
quit;

* Get return at month prior to announcement;
proc sql;
	create table ibescrspcomp2 as select a.*, b.RET as RET_1, b.DATE
	from ibescrspcomp1 as a inner join crspmsf_delist as b
	on a.PERMNO = b.PERMNO and 
		a.ANNDATS_ACTUAL >= b.DATE and
		intck('month', b.DATE, a.ANNDATS_ACTUAL) <= 1
	order by GVKEY, PENDS, ANALYS, DATE desc
	;
quit;
* Keep only most recent observation;
proc sort data = ibescrspcomp2 nodupkey;
	by GVKEY PENDS ANALYS; 
run;

* Calculate cumulative return over t-12 to -2;
proc sql;
	create table ibescrspcomp3 as select distinct a.*, (exp(sum(log(1 + b.RET))) - 1) as RET_12_2
	from ibescrspcomp2 as a inner join crspmsf_delist as b
	on a.PERMNO = b.PERMNO and (1 <= intck('month', b.DATE, a.DATE) <= 12)
	group by GVKEY, PENDS, ANALYS
	;
quit;
data ibescrspcomp3;
	set ibescrspcomp3;
	drop DATE;
run;

* Collect stock price on announcement day (closest within two weeks);
proc sql;
	create table ibescrspcomp4 as select a.*, abs(b.PRC) as PRCC, b.DATE
	from ibescrspcomp3 as a inner join crspmsf_delist as b
	on a.PERMNO = b.PERMNO and (0 <= intck('weekday', b.DATE, a.DATADATE) <= 10)
	having missing(b.PRC) = 0
	order by GVKEY, PENDS, ANALYS, DATE desc
	;
quit;
proc sort data = ibescrspcomp4 nodupkey;
	by GVKEY PENDS ANALYS;
run;

* Calculate stock volatility over the past 5 years, left joining;
proc sql;
	create table ibescrspcomp5 as select distinct a.*, std(b.RET)*sqrt(12) as EQUITY_VOL
	from ibescrspcomp4 as a left join crspmsf_delist as b
	on a.PERMNO = b.PERMNO and (0 <= intck('month', b.DATE, a.PENDS) <= 60)
	group by a.PERMNO, PENDS, ANALYS
	;
quit;

* Confirm no duplicates here and drop variables;
proc sort data = ibescrspcomp5 out = ibescrspcomp_final(drop = HSHRCD DATE PDF FPI) nodupkey;
	by GVKEY PENDS ANALYS;
run;



/**********************************************************************************************
* MERGE WRDS RATIO DATASET
**********************************************************************************************/

* Merge with ibescrspcomp, keeping only crsp and ibes variables and ratios of interest;
proc sql;
	create table ibesratios0 as select a.TICKER, a.PENDS, a.ANNDATS_ACTUAL, a.EPS, 
		a.LEAD_PENDS, a.LEAD_ANNDATS_ACTUAL, a.LEAD_EPS, a.ANALYS, a.FCAST, a.ANNDATS, a.PERMNO, 
		a.GVKEY, a.DATADATE, a.T_DATADATE, a.QTR, a.SIC_1, a.SIC_2, a.RET_1, a.RET_12_2, 
		a.LOG_AT, a.LAG_LOG_AT, a.LAG2_LOG_AT, a.PRCC, a.EQUITY_VOL,
		b.*
	from ibescrspcomp_final as a inner join firm_ratio_&freq as b
	on a.GVKEY = b.GVKEY and a.PENDS = b.QDATE
	;
quit;
data ibesratios0;
	set ibesratios0;
	drop QDATE;
run;


/**********************************************************************************************
* MERGE IN LAGS OF EPS 
**********************************************************************************************/

proc sql;
	create table ibesratios as select a.*, b.LAG_EPS, b.LAG2_EPS
	from ibesratios0 as a inner join epslags_&freq as b
	on a.TICKER = b.TICKER and a.PENDS = b.PENDS
	;
quit;



/**********************************************************************************************
* FINAL CLEANING AND OUTPUT
**********************************************************************************************/

* Calculate number of analysts per announcement;
proc sql;
	create table ibesratios as select *, count(ANALYS) as N_ANALYSTS
	from ibesratios
	group by GVKEY, PENDS
	;
quit;

* Take first announcement within each firm-T_DATADATE;
* Note: this will mean youre only keeping the earliest within a year (not quarter)
for quarterly data;
proc sort data = ibesratios out = ids(keep=GVKEY T_DATADATE PENDS) nodupkey;
	by GVKEY T_DATADATE PENDS;
run;
proc sort data = ids nodupkey;
	by GVKEY T_DATADATE;
run;
proc sql;
	create table ibesratios1 as select b.*
	from ids as a inner join ibesratios as b
	on a.GVKEY = b.GVKEY and a.PENDS = b.PENDS
	order by GVKEY, PENDS, ANALYS
	;
quit;

* Winsorize forecasts and realizations normalized by price by year;
data ibesratios1;
	set ibesratios1;
	Y1 = LEAD_EPS/PRCC;
	Y2 = FCAST/PRCC;
run;
%outliers_iqr(input = ibesratios1, output = ibesratios2, var = Y1 Y2, size = 10);
proc means data = ibesratios2 mean min p1 p5 p25 median p75 p95 p99 max skew kurt;
	var Y1 Y2;
run;
data ibesratios2;
	set ibesratios2;
	drop Y1 Y2;
run;

* Do final winsoriation to get rid of extreme forecast errors;
proc sort data = ibesratios2;
	by GVKEY PENDS FCAST;
run;
data errors;
	set ibesratios2;
	FE_MED = (LEAD_EPS - FCAST)/PRCC;
run;
%outliers_iqr(input = errors, output = ibesratios3, var = FE_MED, size = 10);
data ibesratios3;
	set ibesratios3;
	drop FE_MED;
run;
proc sort data = ibesratios3;
	by GVKEY PENDS ANALYS;
run;

* Count number of firm-year observations lost in final cleaning;
proc sort data = ibesratios out = junk nodupkey;
	by GVKEY PENDS;
run;
proc sort data = ibesratios3 out = junk nodupkey;
	by GVKEY PENDS;
run;

* Test for EPS duplicates - there shouldnt be any so these should give the same values;
proc sort data = ibesratios3 out = junk nodupkey;
	by GVKEY PENDS LEAD_EPS;
run;
proc sort data = ibesratios3 out = junk nodupkey;
	by GVKEY PENDS;
run;

* Create firm-year dataset;
proc sort data = ibesratios3 out = ibesratios_firmyear(drop=ANALYS FCAST ANNDATS) nodupkey;
	by GVKEY PENDS;
run;

* Create analyst level dataset deleting firm-specific stuff;
data ibesratios_analyst;
	set ibesratios3;
	keep GVKEY T_DATADATE PENDS LEAD_EPS ANALYS ANNDATS FCAST;
run;

* Download data;
proc download data = ibesratios_analyst out = tds.dataitj_&freq&h;
run;
proc download data = ibesratios_firmyear out = tds.datait_&freq&h;
run;

endrsubmit;

* Check missings and outliers - THERE WILL BE SOME MISSINGS HERE;
proc means data = tds.datait_&freq&h nmiss mean median min p1 p99 max skew kurt; 
run;

%mend form_df;



/**********************************************************************************************
*  Run macro
**********************************************************************************************/
%form_df(freq=a, h=1);
%form_df(freq=a, h=2);
%form_df(freq=a, h=3);
%form_df(freq=a, h=4);
%form_df(freq=a, h=5);
%form_df(freq=q, h=1);
%form_df(freq=q, h=2);
%form_df(freq=q, h=3);
%form_df(freq=q, h=4);


signoff;
