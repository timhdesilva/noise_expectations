from fitrollingML_A import output_MSE as rolling

h = 3
f = 'a'
for m in ['Elastic Net', 'Random Forest']:
    for pdv in [True]:
        if f == 'a' or h <= 4:
            rolling(freq = f, horizon = h, price_div = pdv, method = m, roll_yrs = 0)