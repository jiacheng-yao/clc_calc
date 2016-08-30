library(readxl)
library(BTYD)

setwd('~/Documents/clv_calc/')

transaction_data <- read_excel("FD DE customer orders and revenue (1).xlsx")
transaction_data$order_date <- as.Date(transaction_data$order_date, "%Y%m%d")

end.of.cal.period <- as.Date("2016-02-01")

transaction_data.cal <- transaction_data[which(transaction_data$order_date <= end.of.cal.period), ]

names(transaction_data.cal)[names(transaction_data.cal)=="order_date"] <- "date"
names(transaction_data.cal)[names(transaction_data.cal)=="customer_id"] <- "cust"
names(transaction_data.cal)[names(transaction_data.cal)=="revenue"] <- "sales"

# split.data <- dc.SplitUpElogForRepeatTrans(transaction_data.cal)
# clean.transaction <- split.data$repeat.trans.elog

elog <- transaction_data.cal

elog.ordered <- elog[order(elog$cust, elog$date), ]

unique.custs <- unique(elog.ordered$cust)

first.trans.indices <- rep(0, length(unique.custs))
last.trans.indices <- rep(0, length(unique.custs))
count <- 0
for (i in 1:length(unique.custs)) {
  count <- count + 1
  cust.indices <- which(elog.ordered$cust == unique.custs[i])
  # Of this customer's transactions, find the index of the first one
  first.trans.indices[count] <- cust.indices[1]
  
  # Of this customer's transactions, find the index of the last one
  last.trans.indices[count] <- cust.indices[length(cust.indices)]
}

repeat.trans.elog <- elog.ordered[-first.trans.indices, ]

first.trans.data <- elog.ordered[first.trans.indices, ]
last.trans.data <- elog.ordered[last.trans.indices, ]


# [-1] is because we don't want to change the column name for custs
names(first.trans.data)[-1] <- paste("first.", names(first.trans.data)[-1], sep = "")
names(first.trans.data)[which(names(first.trans.data) == "first.date")] <- "birth.per"
names(last.trans.data) <- paste("last.", names(last.trans.data), sep = "")

# [-1] is because we don't want to include two custs columns
cust.data <- data.frame(first.trans.data, last.trans.data[, -1])
names(cust.data) <- c(names(first.trans.data), names(last.trans.data)[-1])


freq.cbt <- dc.CreateFreqCBT(transaction_data.cal)