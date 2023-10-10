* This Macro returns a dataset of EA dates identified following the Johnson and So (2018) JAR
methodology for all firms with announcement days between startyr and endyr. This Macro must
be run on the WRDS server and have already specified the home directory location as home;

%macro get_EAs(dataout=, startyr=, endyr=);
/**********************************************************************************************
*  Get EAs from IBES
**********************************************************************************************/

* Download IBES analyst data, and keep a list of earnings announcements where
there was at least one analyst forecast within the 90 days leading up to it;
proc sql;
	create table ibes1 as select *
	from ibes.det_epsus
	where (&startyr <= year(ANNDATS) <= &endyr) and USFIRM = 1 and missing(VALUE) = 0 and
		(0 <= intck('day', ANNDATS, ANNDATS_ACT) <= 90)
	order by TICKER, FPEDATS, ANNDATS_ACT desc, ANNDATS desc;
quit;
proc sort data = ibes1 out = ibes2(keep = TICKER ANNDATS_ACT ANNTIMS_ACT) nodupkey;
	by TICKER ANNDATS_ACT;
run;

* Merge with ICLINK table to get PERMNOs, keeping only the best scores (see get_iclink.sas
for detail on scores);
proc sql;
	create table ibes3 as select a.*, b.PERMNO, b.SCORE as IBES_SCORE
	from ibes2 as a inner join home.iclink as b
	on a.TICKER = b.TICKER
	where (0 <= b.SCORE <= 5)
	order by TICKER, ANNDATS_ACT, IBES_SCORE
	;
quit;
proc sort data = ibes3 out = ibes3 nodupkey;
	by TICKER ANNDATS_ACT;
run;

* Delete small number (approximately 150) observations that have different IBES tickers
associated to the same PERMNO (not sure why this occurs);
proc sort data = ibes3 uniqueout = ibes4 nouniquekey;
	by PERMNO ANNDATS_ACT;
run;

* Get GVKEYs from CRSP-COMPUSTAT linking table;
proc sql;
	create table ibes5 as select a.*, b.UGVKEY
	from ibes4 as a inner join crspa.ccmxpf_lnkused as b
	on a.PERMNO = b.UPERMNO and
		b.ULINKTYPE in ('LC', 'LU') and
 		b.USEDFLAG = 1 and
		(b.ULINKDT <= a.ANNDATS_ACT or b.ULINKDT = .B) and
		(b.ULINKENDDT >= a.ANNDATS_ACT or b.ULINKENDDT = .E) and
		b.ULINKPRIM in ('P', 'C')
	order by TICKER, ANNDATS_ACT
	;
quit;

* Should be no duplicates in either of these;
proc sort data = ibes5 out = ibes5 nodupkey;
	by PERMNO ANNDATS_ACT;
run;
proc sort data = ibes5 out = ibes5 nodupkey;
	by UGVKEY ANNDATS_ACT;
run;



/**********************************************************************************************
*  Get COMPUSTAT EA dates
**********************************************************************************************/

* Desired variables from COMPUSTAT;
%let variables =  GVKEY DATADATE INDFMT POPSRC CONSOL DATAFMT TIC CONM RDQ DATACQTR FYEARQ;

* Data extraction step;
data compustat1;
   	set compa.fundq (keep = &variables);
   	where DATAFMT = 'STD' and POPSRC = 'D' and INDFMT = 'INDL' and CONSOL ='C' and
		(&startyr <= year(RDQ) <= &endyr) ;
	drop INDFMT POPSRC CONSOL DATAFMT;
run;

* Delete duplicates from multiple reports, taking EARLIEST report for each DATADATE;
proc sort data = compustat1;
	by GVKEY DATADATE RDQ;
run;
proc sort data = compustat1 nodupkey;
	by GVKEY DATADATE;
run;

* Delete duplicate observations due to multiple DATADATES with same report date, taking
the latest DATADATE;
proc sort data = compustat1;
	by GVKEY RDQ descending DATADATE;
run;
proc sort data = compustat1 nodupkey;
	by GVKEY RDQ;
run;



/**********************************************************************************************
* MERGE COMPUSTAT DATA WITH IBES
**********************************************************************************************/

* Merge COMPUSTAT and IBES, keeping only observations that are within 2 week days, and
choosing the earlier dates as the earnings announcement date;
proc sql;
	create table ealist1 as select *,
		min(a.ANNDATS_ACT, b.RDQ) as EA_DATE format = date9.
	from ibes5 as a inner join compustat1 as b
	on a.UGVKEY = b.GVKEY and
		(-2 <= intck('weekday', a.ANNDATS_ACT, b.RDQ) <= 2)
	order by PERMNO, RDQ, ANNDATS_ACT
	;
quit;

* Delete duplicate firm-quarters that are due to having multiple IBES observations that
match with the same RDQ date (not sure why this is), keeping the earliest IBES announcement date;
proc sort data = ealist1 nodupkey;
	by PERMNO RDQ;
run;

* Delete duplicate firm-quarters that come from multiple COMPUSTAT DATADATEs having
similar report dates (within 2 days of ANNDATS_ACT), keeping the latest one.
Reason for keeping later DATADATE is probably more accurate.;
proc sort data = ealist1;
	by PERMNO ANNDATS_ACT descending DATADATE;
run;
proc sort data = ealist1 nodupkey;
	by PERMNO ANNDATS_ACT;
run;

* If IBES is the same date as the chosen date and the time stamp occured after the close,
then move the EA date back one date;
data ealist1;
	set ealist1;
	AFTER_CLOSE = (ANNTIMS_ACT >= '16:00:00't);
	if AFTER_CLOSE = 1 and ANNDATS_ACT = EA_DATE
		then EA_DATE = intnx('weekday', EA_DATE, 1);
	drop AFTER_CLOSE ANNTIMS_ACT;
run;

* Rank earnings announcements;
data ealist1;
	set ealist1;
	if missing(DATACQTR) then delete;
	keep PERMNO GVKEY EA_DATE DATACQTR FYEARQ;
run;
proc sort data = ealist1 out = &dataout;
	by DATACQTR EA_DATE;
run;



%mend get_EAs;
