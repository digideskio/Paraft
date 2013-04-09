/********************************************************************************
** Form generated from reading UI file 'cameraanimator.ui'
**
** Created: Wed Jan 23 20:22:38 2013
**      by: Qt User Interface Compiler version 4.8.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CAMERAANIMATOR_H
#define UI_CAMERAANIMATOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QListWidget>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CANWidget
{
public:
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QLabel *label;
    QDoubleSpinBox *tspin;
    QLabel *label_2;
    QDoubleSpinBox *pspin;
    QLabel *label_3;
    QDoubleSpinBox *ux;
    QDoubleSpinBox *uy;
    QDoubleSpinBox *uz;
    QLabel *label_4;
    QDoubleSpinBox *lx;
    QDoubleSpinBox *ly;
    QDoubleSpinBox *lz;
    QLabel *label_5;
    QDoubleSpinBox *px;
    QDoubleSpinBox *py;
    QDoubleSpinBox *pz;
    QLabel *label_6;
    QDoubleSpinBox *dtspin;
    QHBoxLayout *horizontalLayout;
    QPushButton *apply;
    QPushButton *deselect;
    QPushButton *reset;
    QListWidget *listWidget;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *deleteframe;
    QPushButton *insertbutton;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *movedown;
    QPushButton *moveup;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *playbutton;
    QPushButton *stopbutton;
    QPushButton *playtoframe;
    QSpinBox *endframe;

    void setupUi(QWidget *CANWidget)
    {
        if (CANWidget->objectName().isEmpty())
            CANWidget->setObjectName(QString::fromUtf8("CANWidget"));
        CANWidget->resize(490, 833);
        verticalLayout = new QVBoxLayout(CANWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(CANWidget);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        tspin = new QDoubleSpinBox(CANWidget);
        tspin->setObjectName(QString::fromUtf8("tspin"));

        gridLayout->addWidget(tspin, 0, 1, 1, 1);

        label_2 = new QLabel(CANWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        pspin = new QDoubleSpinBox(CANWidget);
        pspin->setObjectName(QString::fromUtf8("pspin"));

        gridLayout->addWidget(pspin, 1, 1, 1, 1);

        label_3 = new QLabel(CANWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        ux = new QDoubleSpinBox(CANWidget);
        ux->setObjectName(QString::fromUtf8("ux"));
        ux->setDecimals(8);
        ux->setMinimum(-9001);
        ux->setMaximum(9001);
        ux->setSingleStep(1e-06);

        gridLayout->addWidget(ux, 2, 1, 1, 1);

        uy = new QDoubleSpinBox(CANWidget);
        uy->setObjectName(QString::fromUtf8("uy"));
        uy->setDecimals(8);
        uy->setMinimum(-9001);
        uy->setMaximum(9001);
        uy->setSingleStep(1e-06);

        gridLayout->addWidget(uy, 2, 2, 1, 1);

        uz = new QDoubleSpinBox(CANWidget);
        uz->setObjectName(QString::fromUtf8("uz"));
        uz->setDecimals(8);
        uz->setMinimum(-9001);
        uz->setMaximum(9001);
        uz->setSingleStep(1e-06);

        gridLayout->addWidget(uz, 2, 3, 1, 1);

        label_4 = new QLabel(CANWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        lx = new QDoubleSpinBox(CANWidget);
        lx->setObjectName(QString::fromUtf8("lx"));
        lx->setDecimals(8);
        lx->setMinimum(-9001);
        lx->setMaximum(9001);
        lx->setSingleStep(1e-06);

        gridLayout->addWidget(lx, 3, 1, 1, 1);

        ly = new QDoubleSpinBox(CANWidget);
        ly->setObjectName(QString::fromUtf8("ly"));
        ly->setDecimals(8);
        ly->setMinimum(-9001);
        ly->setMaximum(9001);
        ly->setSingleStep(1e-06);

        gridLayout->addWidget(ly, 3, 2, 1, 1);

        lz = new QDoubleSpinBox(CANWidget);
        lz->setObjectName(QString::fromUtf8("lz"));
        lz->setDecimals(8);
        lz->setMinimum(-9001);
        lz->setMaximum(9001);
        lz->setSingleStep(1e-06);

        gridLayout->addWidget(lz, 3, 3, 1, 1);

        label_5 = new QLabel(CANWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        px = new QDoubleSpinBox(CANWidget);
        px->setObjectName(QString::fromUtf8("px"));
        px->setDecimals(8);
        px->setMinimum(-9001);
        px->setMaximum(9001);
        px->setSingleStep(1e-06);

        gridLayout->addWidget(px, 4, 1, 1, 1);

        py = new QDoubleSpinBox(CANWidget);
        py->setObjectName(QString::fromUtf8("py"));
        py->setDecimals(8);
        py->setMinimum(-9001);
        py->setMaximum(9001);
        py->setSingleStep(1e-06);

        gridLayout->addWidget(py, 4, 2, 1, 1);

        pz = new QDoubleSpinBox(CANWidget);
        pz->setObjectName(QString::fromUtf8("pz"));
        pz->setDecimals(8);
        pz->setMinimum(-9001);
        pz->setMaximum(9001);
        pz->setSingleStep(1e-06);

        gridLayout->addWidget(pz, 4, 3, 1, 1);

        label_6 = new QLabel(CANWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 0, 2, 1, 1);

        dtspin = new QDoubleSpinBox(CANWidget);
        dtspin->setObjectName(QString::fromUtf8("dtspin"));

        gridLayout->addWidget(dtspin, 0, 3, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        apply = new QPushButton(CANWidget);
        apply->setObjectName(QString::fromUtf8("apply"));

        horizontalLayout->addWidget(apply);

        deselect = new QPushButton(CANWidget);
        deselect->setObjectName(QString::fromUtf8("deselect"));

        horizontalLayout->addWidget(deselect);

        reset = new QPushButton(CANWidget);
        reset->setObjectName(QString::fromUtf8("reset"));

        horizontalLayout->addWidget(reset);


        verticalLayout->addLayout(horizontalLayout);

        listWidget = new QListWidget(CANWidget);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));

        verticalLayout->addWidget(listWidget);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        deleteframe = new QPushButton(CANWidget);
        deleteframe->setObjectName(QString::fromUtf8("deleteframe"));

        horizontalLayout_2->addWidget(deleteframe);

        insertbutton = new QPushButton(CANWidget);
        insertbutton->setObjectName(QString::fromUtf8("insertbutton"));

        horizontalLayout_2->addWidget(insertbutton);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        movedown = new QPushButton(CANWidget);
        movedown->setObjectName(QString::fromUtf8("movedown"));

        horizontalLayout_3->addWidget(movedown);

        moveup = new QPushButton(CANWidget);
        moveup->setObjectName(QString::fromUtf8("moveup"));

        horizontalLayout_3->addWidget(moveup);


        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        playbutton = new QPushButton(CANWidget);
        playbutton->setObjectName(QString::fromUtf8("playbutton"));

        horizontalLayout_4->addWidget(playbutton);

        stopbutton = new QPushButton(CANWidget);
        stopbutton->setObjectName(QString::fromUtf8("stopbutton"));

        horizontalLayout_4->addWidget(stopbutton);

        playtoframe = new QPushButton(CANWidget);
        playtoframe->setObjectName(QString::fromUtf8("playtoframe"));

        horizontalLayout_4->addWidget(playtoframe);

        endframe = new QSpinBox(CANWidget);
        endframe->setObjectName(QString::fromUtf8("endframe"));
        endframe->setMinimum(1);
        endframe->setMaximum(1000000);

        horizontalLayout_4->addWidget(endframe);


        verticalLayout->addLayout(horizontalLayout_4);


        retranslateUi(CANWidget);

        QMetaObject::connectSlotsByName(CANWidget);
    } // setupUi

    void retranslateUi(QWidget *CANWidget)
    {
        CANWidget->setWindowTitle(QApplication::translate("CANWidget", "Camera Animator Options", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("CANWidget", "Time:", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("CANWidget", "Pause:", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("CANWidget", "Up:", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("CANWidget", "Look:", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("CANWidget", "Position:", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("CANWidget", "Default T:", 0, QApplication::UnicodeUTF8));
        apply->setText(QApplication::translate("CANWidget", "Apply", 0, QApplication::UnicodeUTF8));
        deselect->setText(QApplication::translate("CANWidget", "De-Select", 0, QApplication::UnicodeUTF8));
        reset->setText(QApplication::translate("CANWidget", "Reset", 0, QApplication::UnicodeUTF8));
        deleteframe->setText(QApplication::translate("CANWidget", "Delete Key Frame", 0, QApplication::UnicodeUTF8));
        insertbutton->setText(QApplication::translate("CANWidget", "Insert Key Frame", 0, QApplication::UnicodeUTF8));
        movedown->setText(QApplication::translate("CANWidget", "Move Down", 0, QApplication::UnicodeUTF8));
        moveup->setText(QApplication::translate("CANWidget", "Move Up", 0, QApplication::UnicodeUTF8));
        playbutton->setText(QApplication::translate("CANWidget", "Play", 0, QApplication::UnicodeUTF8));
        stopbutton->setText(QApplication::translate("CANWidget", "Stop", 0, QApplication::UnicodeUTF8));
        playtoframe->setText(QApplication::translate("CANWidget", "Play To Frame", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class CANWidget: public Ui_CANWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CAMERAANIMATOR_H
