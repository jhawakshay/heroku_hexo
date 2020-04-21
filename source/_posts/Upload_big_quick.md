---
title: Quick fix to upload Big files in RShiny
date: 2019-10-12 21:58:47
tags: [R Shiny, Data Visualization,Hack, Fix, data.table, R Package]
---

## Upload Big and Quick

In this post I am going to provide you a quick fix to upload big files _(in Gigabytes)_ within seconds. The pre-requisites for these are very basic; I am assuming one know a little RShiny and have written codes on R.

In RShiny, there is a widget **File Input** which is used to upload files from your computer to your Dashboard so that one can do analysis on the same. The widget is a very easy to use with a click and selecting the file but the tricky part about it is _LIMIT SIZE of FILE_. One cannot upload a file of size more than 5MB.

**YES! THIS MUCH ONLY**

While I was working on one of the Dashboard, I was dealing with Gigabytes of Data and faced the first hurdle. I couldn't step ahead before fixing this and did a lot of research around the same but couldn't find a solution.

As Data Scientists or Data Analysts, one needs to find a solution to a problem. Either by hook or crook! Just Kidding!
I am a big fan of data.table package in R. With a single use of this package, my Data wrangling & Analysis becomes really super faster with just few lines of codes. Without wasting much of your time, here is the fix.

**File Input Widget**
![File imported successfuly when size is 5MB or less](/images/datatable/Image_Upload_complete.JPG)

As soon as I increase the file size, I get the following error.
![File imported successfuly when size is 5MB or less](/images/datatable/Image_Upload_Error.JPG)

**Use fread to read quick and fast**
In this fix, I have not used the File Input widget, instead divided the process into three steps. 
1. Place a text input to copy the location
1. Place a clickable button and a dropdown to select the file to upload
1. After fetching the location and file name, reading the file with **fread from datatable package**


**R Code**
_Install packages_
```

#### Packages to install ###
### Create a List ###
packages     <- c("data.table", "shinydashboard", "shiny", "htmltools", "DT" , "dplyr")

### Install the packages ###
if(length(setdiff(packages,rownames(installed.packages())))>0){
  install.packages(setdiff(packages,rownames(installed.packages())), dependencies = T)
}

### Load the packages in R ###
sapply(packages, require, character.only =T)


### For handling error on Shiny app ###
options(shiny.sanitize.errors=TRUE)

```
_Create inital Dashboard design_
Here I have used text input where the file location will be copied. Once this is done, click button **Find Files** which will list all the files in the dropdown. Now, all one has to do is select the file which is to be imported.

```
UI <-
  dashboardPage(
    dashboardHeader(title = h4("Import Big and Quick")),
    dashboardSidebar(
      sidebarMenu(

        ###############################################
        ######    Adding the Data Import Section  #####
        menuItem("Import Data", tabName = "DataImport",
                 icon = icon("table"),startExpanded = FALSE
        ),width = 250
      )),

    dashboardBody(
      tabItems(
        tabItem(
          fluidRow(h1("Import Data")),
          tags$hr(),
          fluidRow(
            column(3, textInput("location", "Copy the location to upload Data"),
                   fluidRow(
                     column(12,actionButton("goButton", "Find Files")),
                     br(),
                     column(12,uiOutput("filenames")))),
            tags$style(type="text/css",
                       ".shiny-output-error { visibility: hidden; }",
                       ".shiny-output-error:before { visibility: hidden; }"
            ),
            column(9,DT::dataTableOutput("Dev.table"))
          )
        , tabName = "DataImport"))
    ))   
```

**First View: Showing the elements**
![First View](/images/datatable/First_Look.JPG)

**Second View: Copy the location and find files**
![Second look of the Dashboard](/images/datatable/Second_Look.JPG)

**Third View: Select the file to be imported**
![Third look of the Dashboard](/images/datatable/Third_Look.JPG)

**Fourth View: Upload Progress Indicator**
![Fourth look of the Dashboard](/images/datatable/Fourth_Look.JPG)

**Fifth View: Showcase Uploaded Data**
![Fifth look of the Dashboard](/images/datatable/Fifth_Look.JPG)


**Server Function**
```
server <- function(input, output, session){

  ####################################################
  #       Reactive Function for the location         #
  nlocation         <- eventReactive(input$goButton,{
    loc             <- input$location
    loc.re          <- gsub("\\\\", "/", readClipboard())
    return(loc.re)
  })

  print("Waiting user to enter the location")

  #####################################################
  # This snippet gives you the files in the location #
  files.lisitng                  <- reactive({
    list.file              <-list.files(path = nlocation(),pattern = "*.csv")
    print(glimpse(nlocation()))
    return(list.file)
  })

  ################# Dropdown for Development Data  ###################
  output$filenames               <- renderUI({
    selectInput("filename", "Select file to upload into tool:", choices = c(" ",files.lisitng()), selected = NULL, multiple = F)
  })

  print("Getting the list of all the csv files")

  ################################################################
  # Read file as soon as we getinput from dropdown widget
  print("Reading the file into R")

  # This is a check to see if the user has selected a file or not
  f                               <- reactive(nchar(input$filename))

  data_                            <- reactive({
    if(f()>2){
      location.f                     <-  nlocation()
      file.name                      <-  as.character(input$filename)
      print(glimpse(input$filename))


      withProgress(message = 'Reading Development Data',
                   detail = 'This may take a while....', {
                     N   <- 20
                     for(i in 1:N){

                       # Update progress
                       incProgress(1/i)

                       # Long Running Task
                       Sys.sleep(0.7)
                     }
                     file.upload                    <-  fread(paste0(location.f,"/",file.name))
                   })

    }else{return()}
  })


  output$Dev.table   <- DT::renderDataTable(
    head(data_(),200))


}

shinyApp(UI, server)
```

**Comparison of read.csv and fread**
```
################################################################
## Comparison of read.csv and fread

loc <- "E:/CV/Startups/InCred/Model Building Sample"


system.time(file.read <- read.csv(file = "E:/CV/Startups/InCred/Model Building Sample/full_data.csv"))
system.time(file.read <- fread(file = "E:/CV/Startups/InCred/Model Building Sample/full_data.csv"))

object.size(file.read)

```
![Time Difference in read.csv & fread](/images/datatable/Console_time_difference.JPG)

![fread: As fast as Usain Bolt](/images/datatable/Usain_Bolt.JPG)

Until then, Keep uploading Big and Quick and do use data.table. It is an awesome package in R.</br>
Please write to me on akshayjhawar.nitj@gmail.com for any further queries
