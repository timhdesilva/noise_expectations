%macro createlags(data=, idvar=, datevar=, vars=, nlags=1);

%let n=%sysfunc(countw(&vars));

proc sort data = &data;
	by &idvar &datevar;
run;

data &data;
	set &data;
	by &idvar;
	%do i=1 %to &n;
		%let var = %scan(&vars,&i);
		%do j=1 %to &nlags;
			%if &j=1 %then
				%do;
					LAG_&var = ifn(first.&idvar=0, lag&j(&var), .);
				%end;
			%else
				%do;
					LAG&j._&var = ifn(lag&j(&idvar) = &idvar, lag&j(&var), .);
				%end;
		%end;
	%end;
run;
%mend;
