%macro get_crspmsf_delist(outdata=);

data delists;
    set crspa.msedelist;
    if missing(DLRET)
        and DLSTCD < 600 and DLSTCD >= 400
        then DLRET = -.3;
    if missing(DLRET) then delete;
    keep PERMNO DLSTDT DLRET;
run;

proc sql;
    create table crspmsf_delist0 as select a.*, b.dlret
    from crspa.msf as a left join delists as b
    on a.PERMNO = b.PERMNO and
        intnx('month', a.DATE, 0, 'END') = intnx('month', b.DLSTDT, 0, 'END')
    order by PERMNO, DATE
    ;
quit;

data crspmsf_delist0;
    set crspmsf_delist0;
    if missing(RET) then RET_ADJ = DLRET;
    else RET_ADJ = (1+RET)*(1+DLRET)-1;
    drop DLRET;
run;

data &outdata;
    set crspmsf_delist0;
    if missing(RET_ADJ) = 0 then
        do;
            RET = RET_ADJ;
            DELIST = 1;
        end;
    else DELIST = 0;
    drop RET_ADJ;
run;

%mend;
