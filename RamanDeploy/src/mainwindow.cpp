#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtCharts>
#include <QHBoxLayout>
#include <QLayoutItem>
#include <QGridLayout>
#include <QLabel>
#include <load_onnx.h>

// QT_CHARTS_USE_NAMESPACE

int datasetIndex = 0;   // Dataset subscript
int classNumber = 0;    // classification

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

QStringList labelName;
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

    qDebug()<<min;
    QList<float> output_data;
    for(int i=0;i<data.length();i++) {
        double normalized_value = (data[i] - min) / (max - min);
        output_data.append(normalized_value);
    }
    return output_data;
}


// Import the data and test
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
    QList<float> spectrumList;
    for(int i = 1 ; i < fCount; i++)
    {
        QStringList tempbar = tempOption.at(i).split(",");
        if(i == 1) {
            for(int j=0; j<tempbar.length(); j++) {
                raman_shift.append(tempbar[j].toFloat());
            }
        }

        // Here we control which spectral value is selected
        if(i == 2) {
            for(int j=0; j<tempbar.length(); j++) {
                spectrumList.append(tempbar[j].toFloat());
            }
        }
    }

    // normalization
    spectrumList = normalizeData(spectrumList);

    // Creating red lines
    QLineSeries *pLineSeries = new QLineSeries();
    pLineSeries->setPen(QPen(Qt::red, 1, Qt::SolidLine));
    for(int i=0; i<raman_shift.length();i++) {
        float x = raman_shift[i];
        float y = spectrumList[i];
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

    //    FILE* f1 =fopen("covid.onnx","rb");
    //    fclose(f1);
    // resnet50
    load_onnx *loadonnx = new load_onnx();

    QString modelPath ="Model/" + comboxDataset + "/" + comboxModel + ".onnx";
    int classesPath = comboxDataset.mid(comboxDataset.length()-2, 1).toInt();
    QList<float> onnx_result = loadonnx->inferenceModel(spectrumList, modelPath, classesPath);

    // Display the final inference result
    qDebug()<<onnx_result[0];
    for(int i=0; i<progressList.length(); i++){
        float output = onnx_result[i] * 100;
        progressList[i]->setValue(output);
        QString number = QString::number(output, 'f', 2);
        radioList[i]->setText(QString(labelName[i] +"("+number+"%)"));
    }
}
