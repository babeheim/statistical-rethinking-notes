data {
  int<lower=1> n_observations;
  int<lower=1> n_monasteries;
  int<lower=1> n_days[n_observations];
  int<lower=0> n_books[n_observations];
  int<lower=1,upper=n_monasteries> monastery[n_observations];
}
parameters {
  real log_lambda[n_monasteries];
  real logit_pr_rest[n_monasteries];
}
model {
  log_lambda ~ normal(0, 1);
  logit_pr_rest ~ normal(-1, 1);
  for (i in 1:n_observations) {
    // marginalize over the possible days worked
    real log_prob_day[n_days[i]];
    for (k in 1:n_days[i]) {
      real alpha = log(k) + log_lambda[monastery[i]];
      log_prob_day[k] = binomial_lpmf(k | n_days[i], 1 - inv_logit(logit_pr_rest[monastery[i]])) + poisson_log_lpmf(n_books[i] | alpha);
    }
    real log_sum_joint = log_sum_exp(log_prob_day);
    if (n_books[i] == 0) log_sum_joint = log_sum_exp(log_sum_joint, binomial_lpmf(0 | n_days[i], 1 - inv_logit(logit_pr_rest[monastery[i]])));
    target += log_sum_joint;
  }
}

