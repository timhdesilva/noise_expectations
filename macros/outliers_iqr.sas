%macro outliers_iqr(input=, output=, var= , size=5);

%let Q1=;
%let Q3=;
%let varL=;
%let varH=;

%let n=%sysfunc(countw(&var));
%do i= 1 %to &n;
%let val = %scan(&var,&i);
%let Q1 = &Q1 &val._P25;
%let Q3 = &Q3 &val._P75;
%let varL = &varL &val.L;
%let varH = &varH &val.H;
%end;

/* Calculate the quartiles and inter-quartile range using proc univariate */
proc means data=&input nway noprint;
var &var;
output out=temp P25= P75= / autoname;
run;

/* Extract the upper and lower limits into macro variables */
data temp;
set temp;
ID = 1;
array varb(&n) &Q1;
array varc(&n) &Q3;
array lower(&n) &varL;
array upper(&n) &varH;
do i = 1 to dim(varb);
lower(i) = varb(i) - &size * (varc(i) - varb(i));
upper(i) = varc(i) + &size * (varc(i) - varb(i));
end;
drop i _type_ _freq_;
run;

data temp1;
set &input;
ID = 1;
run;

data &output;
merge temp1 temp;
by ID;
array var(&n) &var;
array lower(&n) &varL;
array upper(&n) &varH;
do i = 1 to dim(var);
if not missing(var(i)) then do;
if var(i) >= lower(i) and var(i) <= upper(i);
end;
end;
drop &Q1 &Q3 &varL &varH ID i;
run;
%mend;
