# Install and load required packages
# install.packages("tempdisagg")
# install.packages("zoo")
# install.packages("readr")
# install.packages("dplyr")

library(tempdisagg)
library(zoo)
library(readr)
library(dplyr)

# Set working directory
setwd("/Users/julbs/Desktop/nowcasting")

# Load data
data <- read_csv("employment.csv")

# Convert 'month_id' to yearmon
data <- data %>%
  mutate(month_id = as.yearmon(month_id, "%Y-%m"))

# Split into two parts: interpolate before 2021, keep actual from 2021 onward
cutoff <- as.yearmon("2021-01")
data_before <- data %>% filter(month_id < cutoff)
data_after  <- data %>% filter(month_id >= cutoff)

# Step 1: Interpolate quarterly values (from data_before only)
quarterly_data <- data_before %>%
  filter(!is.na(unemployment)) %>%
  select(month_id, unemployment)

# Convert to quarterly ts
ts_quarterly <- ts(
  quarterly_data$unemployment,
  start = c(as.numeric(format(as.Date(as.yearmon(min(quarterly_data$month_id))), "%Y")),
            (as.numeric(format(as.Date(as.yearmon(min(quarterly_data$month_id))), "%m")) - 1) / 3 + 1),
  frequency = 4
)

# Run Denton interpolation using "first" conversion
td_result <- td(ts_quarterly ~ 1, to = "monthly", conversion = "first")
predicted_series <- zoo(predict(td_result), as.yearmon(time(predict(td_result))))

# Assign interpolated values to data_before
data_before <- data_before %>%
  mutate(unemployment_monthly = coredata(predicted_series[month_id]))

# Step 2: Use actual unemployment for data_after
data_after <- data_after %>%
  mutate(unemployment_monthly = unemployment)

# Step 3: Combine and save
final_data <- bind_rows(data_before, data_after) %>%
  arrange(month_id) %>%
  mutate(month_id = format(month_id, "%Y-%m"))  # Optional: convert back to character

write_csv(final_data, "employment_monthly.csv")
