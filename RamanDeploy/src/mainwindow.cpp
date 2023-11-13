#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtCharts>
#include <QHBoxLayout>
#include <QLayoutItem>
#include <QGridLayout>
#include <QLabel>
#include <QApplication>
#include <QTextStream>
#include <QFile>
#include <QDateTime>
#include <load_onnx.h>

// QT_CHARTS_USE_NAMESPACE

int datasetIndex = 0;   // Dataset subscript
int classNumber = 0;    // classification
QStringList labelName;  // label name

QStringList classes;    // classification name
int number = 0; // number of classifications

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    hblayout = ui->horizontalLayout;
    ui->comboBox->setCurrentIndex(0);
    on_comboBox_activated(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_comboBox_activated(int index)
{
    //1. Select the dataset
    datasetIndex = ui->comboBox->currentIndex();
    labelName.clear();
    if(datasetIndex == 0)
    {
        //pnas_dataset(2)
        classNumber = 2;
        labelName<<"HN"<<"CVB";

    }else if(datasetIndex ==1)
    {
        //ovarian_cancer(2)
        classNumber = 2;
        labelName<<"Healthy"<<"Ovarian";

    }else if(datasetIndex == 2)
    {
        //covid-19(3)
        classNumber = 3;
        labelName << "Covid"<<"Health"<<"Suspected";

    }else if(datasetIndex == 3)
    {
        //cell_spectrum(5)
        classNumber = 5;
        labelName<<"MC"<<"PM"<<"NSF"<<"PMC"<<"TAF";

    }else if(datasetIndex == 4)
    {
        //single_cell(6)
        classNumber = 6;
        labelName<<"CLP"<<"CMP"<<"HSC"<<"MPP1"<<"MPP2"<<"MPP3";
    }else if(datasetIndex == 5)
    {
        //new dataset
        classNumber = number;
        labelName = classes;
    }

    // 2.Generate the interface
    // Clear the controls inside the layout
    clearLayout(hblayout);
    progressList.clear();
    radioList.clear();
    // Generating new controls
    for(int i=0; i<classNumber; i++)
    {
        createResultsUI(hblayout, labelName[i]);
    }
}

// Clear all controls inside the layout
void MainWindow::clearLayout(QLayout *layout)
{
    QLayoutItem *child;
    if (layout == NULL)
        return;
    while ((child = layout->takeAt(0)) != 0)
    {
        //setParent is NULL to prevent the screen from disappearing after deletion
        if(child->widget())
        {
            child->widget()->setParent(NULL);
            delete child->widget();
        }else if(child->layout())
        {
            clearLayout(child->layout());
        }
        delete child;
        child = NULL;
    }
}

void MainWindow::createResultsUI(QHBoxLayout *hblayout, QString labelName)
{
    QGridLayout *glayout = new QGridLayout();
    QRadioButton *radioButton = new QRadioButton();
    radioButton->setText(labelName);
    radioList.append(radioButton);
    QProgressBar *progressbar = new QProgressBar();
    progressbar->setOrientation(Qt::Vertical);
    progressbar->setMinimum(0);
    progressbar->setMaximum(100);
    progressbar->setValue(25);
    if(classNumber == 2)
    {
        progressbar->setFixedWidth(100);
    }else if(classNumber == 3)
    {
        progressbar->setFixedWidth(80);
    }else if(classNumber == 4)
    {
        progressbar->setFixedWidth(60);
    }else if(classNumber == 5)
    {
        progressbar->setFixedWidth(50);
    }else if(classNumber == 6)
    {
        progressbar->setFixedWidth(30);
    }
    QLabel *label1 = new QLabel("1-");
    QLabel *label2 = new QLabel("0.5-");
    QLabel *label3 = new QLabel("0-");
    glayout->addWidget(radioButton,0,0,1,3);
    glayout->addWidget(label1,1,0);
    glayout->addWidget(label2,2,0);
    glayout->addWidget(label3,3,0);
    glayout->addWidget(progressbar,1,1,3,2);
    hblayout->addLayout(glayout);

    progressList.append(progressbar);
}

// normalize the data
QList<float> MainWindow::normalizeData(QList<float>& data)
{
    float min = *std::min_element(data.begin(), data.end());
    float max = *std::max_element(data.begin(), data.end());

    QList<float> output_data;
    for(int i=0;i<data.length();i++) {
        double normalized_value = (data[i] - min) / (max - min);
        output_data.append(normalized_value);
    }
    return output_data;
}


// Import the data and inference
void MainWindow::on_pushButton_17_clicked()
{
    // Get the currently selected dataset and model
    QString comboxDataset = ui->comboBox->currentText();
    QString comboxModel = ui->buttonGroup->checkedButton()->text();
    // load file
    QFileDialog* fd = new QFileDialog(this);
    QString fileName = fd->getOpenFileName(this,tr("Open File"),"",tr("Excel(*.csv)"));
    if(fileName == "")
        return;
    QDir dir = QDir::current();
    QFile file(dir.filePath(fileName));
    if(!file.open(QIODevice::ReadOnly))
        qDebug()<<"OPEN FILE FAILED";
    QTextStream * out = new QTextStream(&file);
    QStringList tempOption = out->readAll().split("\n");
    int fCount = tempOption.count();
    //    qDebug()<<fCount;
    QList<float> raman_shift;
    QList<QList<float>> spectrumList;
    for(int i = 1 ; i < fCount; i++)
    {
        QStringList tempbar = tempOption.at(i).split(",");
        tempbar = tempbar.mid(2);

        if(i == 1) {
            qDebug()<<tempbar;
            for(int j=0; j<tempbar.length(); j++) {
                raman_shift.append(tempbar[j].toFloat());
            }
        }
        if(i > 1) {
            // Before adding data, make sure you have added enough empty rows to the spectrumList
            if( (i-2) >= spectrumList.size())
            {
                for(int x = spectrumList.size(); x <= (i-2); ++x)
                {
                    QList<float> newRow;
                    spectrumList.append(newRow);
                }
            }
            // Loop through the data for each row
            for(int j=0; j<tempbar.length(); j++) {
                spectrumList[i-2].append(tempbar[j].toFloat());
            }
        }
    }
    // remove the last empty row
    if(!spectrumList.isEmpty()){
        spectrumList.removeLast();
    }

    // normalization
    for(int i=0; i < spectrumList.size(); i++)
    {
        spectrumList[i] = normalizeData(spectrumList[i]);
    }

    // Creating red lines
    QLineSeries *pLineSeries = new QLineSeries();
    pLineSeries->setPen(QPen(Qt::red, 1, Qt::SolidLine));
    for(int i=0; i<raman_shift.length();i++) {
        float x = raman_shift[i];
        float y = spectrumList[0][i];
        pLineSeries->append(x, y);
    }

    pLineSeries->setName(comboxDataset);
    pLineSeries->setColor(Qt::red);

    // creating QChart
    QChart *pChart = new QChart();
    pChart->addSeries(pLineSeries);
    pChart->createDefaultAxes();
    pChart->setTheme(QChart::ChartThemeDark);
    pLineSeries->setColor(Qt::red);
    pChart->axes().at(0)->setTitleText("Raman Shift/cm^-1");
    pChart->axes().at(1)->setTitleText("Intensity");

    // creating QChartView
    MyChartView = ui->graphicsView;
    MyChartView->setChart(pChart);
    MyChartView->setRenderHint(QPainter::Antialiasing);

    file.close(); // Close the file when you're done

    // resnet50
    load_onnx *loadonnx = new load_onnx();

    QString modelPath ="Model/" + comboxDataset + "/" + comboxModel + ".onnx";
    QFile file_model(modelPath);
    if(!file_model.exists())
    {
        qDebug()<<"文件不存在";
        QMessageBox::information(this,"Information","ONNX file does not exist, please check that your file path is correct");
        return;
    }

    // Inference the model output
    int classesPath = comboxDataset.mid(comboxDataset.length()-2, 1).toInt();

    QList<QList<float>> result;

    for(int i=0; i<spectrumList.size(); i++)
    {
        QList<float> onnx_result = loadonnx->inferenceModel(spectrumList[i], modelPath, classesPath);

        // Before adding data, make sure you have added enough empty rows to the result
        if(i >= result.size())
        {
            for(int j = result.size(); j <= i; ++j)
            {
                QList<float> newRow;
                result.append(newRow);
            }
        }
        result[i] = onnx_result;

        // The interface displays only the results of the first row of data
        if(i==0)
        {
            for(int j=0; j<progressList.length(); j++){
                float output = onnx_result[j] * 100;
                progressList[j]->setValue(output);
                QString number = QString::number(output, 'f', 2);
                radioList[j]->setText(QString(labelName[j] +"("+number+"%)"));
            }
        }
    }
    // Display the final inference result
    qDebug()<<result;

    // report generation
    QString directory = "report/";
    QDateTime currentTime = QDateTime::currentDateTime();
    QString reportName = "report_" + comboxDataset + comboxModel + currentTime.toString("yyyyMMdd_hhmmss")+".csv";
    QFile report(directory+reportName);
    if(report.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream stream(&report);
        stream.setFieldWidth(0);
        stream.setFieldAlignment(QTextStream::AlignLeft);
        stream.setPadChar(' ');
        stream.setRealNumberPrecision(16);
        stream.setRealNumberNotation(QTextStream::SmartNotation);

        // classes
        QStringList headers;
        headers << "0" << "1";
        for(int i=0; i<classNumber; i++)
        {
            headers << QString::number(i+2);
        }
        stream << headers.join(",")<<"\n";
        // labels
        labelName.insert(0, "ID");
        labelName << "pre_labels";
        stream<<labelName.join(",")<<"\n";
        // result
        int ID = 1;
        for(const QList<float>& row : result)
        {
            QStringList rowData;
            rowData<<QString::number(ID);
            ID += 1;
            int maxIndex = 0;
            float maxValue = 0;
            for(int i=0;i<row.length();i++) {
                rowData<<QString::number(row[i]);
                if (row[i] > maxValue) {
                    maxValue = row[i];
                    maxIndex = i;
                }
            }
            rowData << QString::number(maxIndex);
            stream << rowData.join(",")<<"\n";
        }
        report.close();
    }

}

// Import a custom model
void MainWindow::on_pushButton_clicked()
{
    QFileDialog* fd = new QFileDialog(this);
    QString filePath = fd->getOpenFileName(this,tr("Open File"),"",tr("ONNX(*.onnx)"));
    if(filePath == "")
        return;
    QFileInfo fileInfo(filePath);
    QString fileName = fileInfo.baseName(); // Gets the file name without a suffix

    ui->radioButton->setChecked(1); // Select the button
    ui->radioButton->setText(fileName); // Change the control text
}

// Import the dataset configuration file
void MainWindow::on_pushButton_2_clicked()
{
    QFileDialog* fd = new QFileDialog(this);
    QString filePath = fd->getOpenFileName(this,tr("Open File"),"",tr("TEXT(*.txt)"));
    if(filePath == "")
        return;
    QDir dir = QDir::current();
    QFile file(dir.filePath(filePath));
    if(!file.open(QIODevice::ReadOnly))
        qDebug()<<"OPEN FILE FAILED";
    QTextStream in(&file);
    QStringList lines;
    while(!in.atEnd())
    {
        QString line = in.readLine();
        lines.append(line);
    }
    file.close();
    // Configuration information
    // the first line is the dataset name
    // the second row is the number of classifications
    // the third line is the classification name
    // (separated by , )
    QString name=lines[0];
    QString strNumber=lines[1];
    number=strNumber.toInt();
    if(number>6 || number<2)
    {
        QMessageBox::information(this,"Warning","The class number of new dataset must between 2 to 6.");
        return;
    }
    classes=lines[2].split(",");
    qDebug()<<classes;
    // Add Item from the drop-down menu and select the newly added item
    ui->comboBox->addItem(name+"("+strNumber+")");
    QMessageBox::information(this,"Hint","The new dataset profile has been successfully imported.");
}


