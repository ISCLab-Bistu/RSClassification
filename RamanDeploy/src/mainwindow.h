#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts>
#include <QHBoxLayout>


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_comboBox_activated(int index);

    void on_pushButton_17_clicked();

    void clearLayout(QLayout *layout);

    void createResultsUI(QHBoxLayout *hblayout, QString labelName);

    QList<float> normalizeData(QList<float>& data);

    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::MainWindow *ui;

    QChartView *MyChartView;   // Canvas objects
    QChart MyChart;             // Chart objects
    QLineSeries MyLineSeries;   // The line object displayed above the chart object
    QVector<QPointF> MyPointf;  // Data needed to draw a line object
    QValueAxis MyAxisX;         // X labels
    QValueAxis MyAxisY;         // Y labels

    QHBoxLayout *hblayout;       // Dynamically generates the layout of the control

    QList<QProgressBar *> progressList;  // Progress bar control pointer
    QList<QRadioButton *> radioList;


};
#endif // MAINWINDOW_H
