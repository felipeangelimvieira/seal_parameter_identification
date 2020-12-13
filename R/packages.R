setwd("~/R")
.libPaths("lib/")

packages.to.install <- c("ggplot2",
                         "data.table",
                         "assertthat",
                         "matlib",
                         "ggthemes",
                         "dplyr",
                         "gridExtra",
                         "jsonlite",
                         "rjson",
                         "cowplot")


install.packages(packages.to.install, dependencies=T)
