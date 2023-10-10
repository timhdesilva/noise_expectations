%macro get_fundq(outdata=, vars=, startyr=, endyr=);

* Desired variables from COMPUSTAT;
%let variables =  &vars GVKEY DATADATE RDQ INDFMT POPSRC CONSOL DATAFMT;

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

* Get CRSP PERMNOs as left join;
proc sql;
	create table linked as select a.*, b.LPERMNO as PERMNO
	from compustat1 as a left join crspa.ccmxpf_linktable as b
	ON 
	/* 1 */ a.GVKEY = b.GVKEY and 
	/* 2 */ b.LINKTYPE in ('LC', 'LU') and
 	/* 3 */ b.USEDFLAG = 1 and 
	/* 4 */ (b.LINKDT <= a.DATADATE or b.LINKDT = .B) and 
    /* 5 */ (b.LINKENDDT >= a.DATADATE or b.LINKENDDT = .E) and
	/* 6 */ b.LINKPRIM in ('P', 'C')
	order by GVKEY, DATADATE;
quit;


data &outdata;
	set linked;
run;


%mend;
