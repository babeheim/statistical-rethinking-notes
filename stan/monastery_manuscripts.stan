data {
  int<lower=1> n_observations;
  int<lower=1> n_monasteries;
  real<lower=0> log_n_days[n_observations];
  int<lower=0> n_books[n_observations];
  int<lower=1,upper=n_monasteries> monastery[n_observations];
}
parameters {
  real log_lambda[n_monasteries];
}
model {
  log_lambda ~ normal(0, 1);
  for (i in 1:n_observations) {
    real alpha = log_n_days[i] + log_lambda[monastery[i]];
    target += poisson_log_lpmf(n_books[i] | alpha);
  }
}

