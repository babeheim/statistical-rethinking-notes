
# Monastic Book Production

A group of monks is copying books. Because books vary in length, and monks work independently we cannot really tell how many books will be finished on a particular day - most days have no books produced, but some days have one book produced, and occasionally even multiple books produced on the same day. Since monks work on books one at a time, we can model the number of books produced on a given day using a Binomial distribution, with the average rate of book production per monk per day as $p$, for $n$ monks.

Assuming $p$ is very small, e.g. 1/365 if it takes 365 days to produce a book, on average, and there are many, many monks working, then this Binomial count will of course become close and closer to following the Poisson distribution with a daily rate $\lambda = \sum_1^n p_i = n p$.

The Poisson is very handy, because you can combine different rates very simply. For example, if one monastary reports daily records of books produced, but a second monastery keeps *weekly* counts, we can calculate the daily rate $\lambda_j$ for each monastary $j$.

```r

library(rethinking)

n_monasteries <- 5
log_lambda <- rnorm(n_monasteries, 0, 1)
n_observations <- 5000
monastery <- sample(seq_len(n_monasteries), n_observations, replace = TRUE)
n_days <- sample(seq_len(5), n_observations, replace = TRUE)
n_books <- rpois(n_observations, n_days * exp(log_lambda[monastery]))

d <- data.frame(
  n_days = n_days,
  n_books = n_books,
  monastery = monastery
)

lambda_emp <- tapply(d$n_books, d$monastery, sum) / tapply(d$n_days, d$monastery, sum)
plot(exp(log_lambda), lambda_emp)
abline(0, 1, lty = 2)

```

How to compare 'apples to apples' here? One strategy is to transform the data to have common units, here "weeks". This could be a complex operation though, if there are many different units of exposure across observations.

Another strategy is to incorporate the unit of exposure into the lambda term itself.

```r

library(rethinking)
library(cmdstanr)

d$log_n_days <- log(d$n_days)

stan_data <- as.list(d)

stan_data$n_observations <- nrow(d)
stan_data$n_monasteries <- length(unique(d$monastery))

m_base <- cmdstan_model("stan/monastery_manuscripts.stan")

fit_base <- m_base$sample(
  data = stan_data, 
  seed = 123, 
  chains = 4, 
  parallel_chains = 4,
  refresh = 500
)

fit_base$output_files() |>
  rstan::read_stan_csv() |>
  extract.samples() -> samples

exp(log_lambda)
apply(exp(samples$log_lambda), 2, mean)
apply(exp(samples$log_lambda), 2, HPDI)

plot(exp(log_lambda), apply(exp(samples$log_lambda), 2, mean))
abline(0, 1, lty = 2)

```


## Incorporating Resting Days

Now we model the idea that on some days, no manuscripts are worked on by anyone. Empirically, we would see a lower rate of production go down, simply because more days are being divided over than actual days worked.

However, we can also decompose this aggregate rate of production into the rate of production on "working" days, and the rate at which "resting days" occur. These are two separate processes, and they have distinct structural features.

Let the rate at which resting days occurs be $p$. Then the number of days worked, out of $n$ days total, is X ~ Binomial(n, 1-p). The rate of manuscript production for these $n$ days is thus X$\lambda$. If we knew what value X was, estimating p would be totally independent from estimating lambda, because we can calculate $Pr(M = m | \lambda, X = k, p)$. But if we don't know X, could we still estimate the true $\lambda$ and $p$?

Yes! Although we don't know the actual value of k for each observation, it turns out we don't need it. Pr(M = m | \lambda, n, p) can be decomposed by the Law of Total Probability.

Yes! We are trying to calculate the log-probability of an observed number of manuscripts given a particular rate and pr_rest. We don't know the number of days actualled owrkd on any given observation of n_days, but we can marginalize over that, since

$$
  \mathrm{Pr}(M = m | \lambda = Xr, n, p) = \sum_{k = 1}^{n} \mathrm{Pr}(M = m | \lambda = kr, n, p) \mathrm{Pr}(X = k | n, p)
$$

Note that although $k$ could equal 0, we don't need to consider it here since $\lambda$ cannot be zero (even though R happily does this).

```r

library(cmdstanr)
library(rethinking)

n_monasteries <- 10
log_lambda <- rnorm(n_monasteries, 0, 1)
logit_p <- rnorm(n_monasteries, -1.0, 1)
n_observations <- 1000
monastery <- sample(seq_len(n_monasteries), n_observations, replace = TRUE)
n_days <- sample(1:5, n_observations, replace = TRUE)
n_days_worked <- rbinom(n_observations, n_days, 1 - logistic(logit_p[monastery]))
n_books <- rpois(n_observations, n_days_worked * exp(log_lambda[monastery]))

d <- data.frame(
  n_days = n_days,
  n_books = n_books,
  monastery = monastery
)

d$log_n_days <- log(d$n_days)

```

Now we try stan code

```r

stan_data <- as.list(d)
stan_data$n_monasteries <- n_monasteries
stan_data$n_observations <- nrow(d)

m_base <- cmdstan_model("stan/monastery_manuscripts.stan")

fit_base <- m_base$sample(
  data = stan_data, 
  seed = 123, 
  chains = 4, 
  parallel_chains = 4,
  refresh = 500
)

m_rest <- cmdstan_model("stan/monastery_manuscripts_rest.stan")

fit_rest <- m_rest$sample(
  data = stan_data, 
  seed = 123, 
  chains = 1, 
  parallel_chains = 1,
  refresh = 500
)

# compare basic model with resting-day model with marginalization

par(mfrow = c(1, 3))

fit_base$output_files() |>
  rstan::read_stan_csv() |>
  extract.samples() -> samples

plot(exp(log_lambda), apply(exp(samples$log_lambda), 2, mean), main = "biased estimates", xlim = c(0, 3), ylim = c(0, 3))
abline(0, 1, lty = 2)

for (i in seq_len(n_monasteries)) {
  true_lambda <- exp(log_lambda)[i]
  est_lambda_lb <- HPDI(exp(samples$log_lambda[,i]))[1]
  est_lambda_ub <- HPDI(exp(samples$log_lambda[,i]))[2]
  lines(c(true_lambda, true_lambda), c(est_lambda_lb, est_lambda_ub))
}

fit_rest$output_files() |>
  rstan::read_stan_csv() |>
  extract.samples() -> samples

plot(exp(log_lambda), apply(exp(samples$log_lambda), 2, mean), main = "unbiased estimates", xlim = c(0, 3), ylim = c(0, 3))
abline(0, 1, lty = 2)

for (i in seq_len(n_monasteries)) {
  true_lambda <- exp(log_lambda)[i]
  est_lambda_lb <- HPDI(exp(samples$log_lambda[,i]))[1]
  est_lambda_ub <- HPDI(exp(samples$log_lambda[,i]))[2]
  lines(c(true_lambda, true_lambda), c(est_lambda_lb, est_lambda_ub))
}

logistic(logit_p)
apply(logistic(samples$logit_pr_rest), 2, mean)

plot(logistic(logit_p), apply(logistic(samples$logit_pr_rest), 2, mean), main = "probability of rest")
abline(0, 1, lty = 2)

for (i in seq_len(n_monasteries)) {
  true_p <- logistic(logit_p)[i]
  est_p_lb <- HPDI(logistic(samples$logit_pr_rest[,i]))[1]
  est_p_ub <- HPDI(logistic(samples$logit_pr_rest[,i]))[2]
  lines(c(true_p, true_p), c(est_p_lb, est_p_ub))
}

```