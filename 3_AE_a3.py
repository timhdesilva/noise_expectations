from fitrollingML_AE import output_MSE as rolling

h = 3
f = 'a'
for m in ['Elastic Net', 'Random Forest', 'Gradient Boosted Tree']:
    for pdv in [True]:
        if f == 'a' or h <= 4:
            rolling(freq = f, horizon = h, price_div = pdv, method = m, roll_yrs = 0)