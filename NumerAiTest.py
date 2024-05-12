from typing import Tuple
import matplotlib.pyplot as plt
from numerapi import NumerAPI
import json
import pandas as pd
import sqlite3
import os
import re
from enum import Enum
from lightgbm import LGBMRegressor
import cloudpickle
import sys
import shutil
import certifi
import urllib3
import logging
from sklearn.ensemble import HistGradientBoostingRegressor
import platform

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)

# import the 2 scoring functions
from numerai_tools.scoring import numerai_corr, correlation_contribution

from PySide6.QtGui import  QGuiApplication, QFont, QFontMetrics, QPainterPath, QColor ,QGradient, QPen, QLinearGradient, QPainter
from PySide6.QtCore import Qt, QPointF, QRectF, QRect ,QDateTime,QCoreApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsItem, QGraphicsSimpleTextItem, QSplitter, QScrollArea,
    QWidget, QListWidget, QTableWidgetItem, QListWidgetItem, QTableWidget, QSpacerItem,
    QProgressBar, QSizePolicy, QAbstractItemView, QSizePolicy, QComboBox,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QLineEdit, QFileDialog,
    QLabel, QPushButton, QMessageBox,QStackedWidget
)
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis, QScatterSeries,
    QSplineSeries,QAreaSeries
)

# Create a class for the main platform UI

class Platform(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        Text_Edit_Style = """
        color: #000000; 
        font-size: 10px; 
        font-family: Arial; 
        background-color: #FFFFFF; 
        border: 1px solid #CCCCCC; 
        border-radius: 5px; 
        padding: 5px;
        selection-background-color: #BFE6FF; 
        selection-color: #000000; 
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000; 
        border-left: 2px solid #000000; 
        border-right: 2px solid #000000; 
        margin: 5px;
        """
        # Create the terminal widget
        self.terminal_widget = QTextEdit()
        self.terminal_widget.setStyleSheet(Text_Edit_Style)
        self.terminal_widget.setReadOnly(True)  # Set read-only property
        
        self.default_folder_path = self.get_default_folder_path()
        self.dynamic_folder_path = None
        
        self.read_parquet_train_file = None
        self.trained_model = None
        self.Validation_Model = None
        
        #Define Feature Set and File Parameters
        self.selected_feature_file = None
        self.selected_feature_set= None
        self.selected_feature_sets_all_features = None
        self.num_of_features = None
        self.live_features_stored = None
         
        self.initialize_all_layouts()
        self.initialize_Column_0_menu_Layout()
        self.initialize_Column_2_Body_model_layout()
   
        #Connect List Widget to NumerAi function
        self.function_Download_all_available_datasets_numerAi()
        # self.function_update_downloaded_datasets_table()
        
        # Create instances of  redirectors
        self.stdout_redirector = StdoutRedirector(text_widget=self.terminal_widget)
        self.stderr_redirector = StderrRedirector(text_widget=self.terminal_widget)
        
        # Redirect sys.stdout and sys.stderr to your redirectors
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
    def initialize_all_layouts(self) -> None:
        """
        Initializes all layout components of the GUI.
        Creates a main horizontal layout and adds four column layouts to it.
        """
        try:
            main_layout = QHBoxLayout()
            self.setLayout(main_layout)

            layout_styles = {
                "Column_0_menu_layout": "#001357",
                "Column_2_Body_model_layout": "#DEDEDC"   
            }
            for layout_name, background_color in layout_styles.items():
                try:
                    layout_name_string = layout_name
                    layout_name = QWidget()  # Create a widget to hold the layout
                    layout_name.setStyleSheet(f"background-color: {background_color};")
                    layout = QVBoxLayout(layout_name)
                    setattr(self, layout_name_string, layout)
                    layout.setContentsMargins(5, 5, 5, 5)
                    main_layout.addWidget(layout_name)  # Add the layout widget to the main layout
                except Exception as e:
                    print(f"Error creating or adding layout {layout_name}: {e}")
        except Exception as e:
            print(f"Error initializing main layout: {e}")
             
    def initialize_Column_0_menu_Layout(self)-> None:
        column_0_layout_internal = QVBoxLayout()
        # Set button styles
        button_style = """
            background-color: #001357;
            color: #F9FFFF;
            font-family: Arial;
             font-size: 18;
        """
        # Create button
        self.button_train_model_step_1 = self.function_create_button("Train", column_0_layout_internal, self.hide_other_layouts, button_style)
        self.button_validate_model_step_2= self.function_create_button("Validate", column_0_layout_internal, self.hide_other_layouts,button_style)
        self.button_peformance_metrics_step_3 = self.function_create_button("Peformance", column_0_layout_internal, self.hide_other_layouts,button_style)
        self.button_live_model_step_4 = self.function_create_button("Muti-Automate", column_0_layout_internal, self.hide_other_layouts,button_style)
        # Add internal layout to the main layout
        self.Column_0_menu_layout.addLayout(column_0_layout_internal)
                       
    def showEvent(self, event):
        super().showEvent(event)
        # Cap the maximum width of the Column_0_menu_layout parent widget to be 15% of the total width
        parent_widget = self.Column_0_menu_layout.parentWidget()
        if parent_widget:
            parent_widget.setMaximumWidth(self.width() * 0.30)
        
    def initialize_Column_2_Body_model_layout(self)-> None:

        try:

            # Create the header widget
            header_widget = QWidget()
            self.Column_2_Body_model_layout.addWidget(header_widget)
            column_2_layout_internal_Header = QVBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #F9FFFF;")
            
            # Create the label for displaying button text
            self.button_label = QLabel("Select button to commence")
            self.button_label.setStyleSheet("font-size: 19px; color: #000000; font-family: Arial;")
            column_2_layout_internal_Header.addWidget(self.button_label)

            # Create the body widget
            self.body_widget = QStackedWidget()
            self.body_widget.setStyleSheet("background-color: #DEDEDC;")
            self.Column_2_Body_model_layout.addWidget(self.body_widget)

            # Create and add different pages to the stacked widget
            self.column_2_layout_internal_train = self.set_layout_to_train()
            self.column_2_layout_internal_Validate = self.set_layout_to_validate()
            self.column_2_layout_internal_Peforomance = self.set_layout_to_peformance()
            self.column_2_layout_internal_Live = self.set_layout_to_Automate()

            self.body_widget.addWidget(self.column_2_layout_internal_train)
            self.body_widget.addWidget(self.column_2_layout_internal_Validate)
            self.body_widget.addWidget(self.column_2_layout_internal_Peforomance)
            self.body_widget.addWidget(self.column_2_layout_internal_Live)

            # Hide all pages initially
            self.body_widget.setCurrentIndex(0)
            
        
            # Create the end widget
            end_widget = QWidget()
            end_widget.setStyleSheet("background-color: #F9FFFF;")
            self.Column_2_Body_model_layout.addWidget(end_widget)
            column_2_layout_internal_End = QVBoxLayout(end_widget)
            column_2_layout_internal_End.addWidget(self.terminal_widget)  # Add terminal widget
            
            # Create a splitter for header and body/end
            splitter_header_body = QSplitter(Qt.Vertical)
            splitter_header_body.addWidget(header_widget)
            splitter_header_body.addWidget(self.body_widget)

            # Create a main splitter for header/body and end
            main_splitter = QSplitter(Qt.Vertical)
            main_splitter.addWidget(splitter_header_body)
            main_splitter.addWidget(end_widget)

            # Set sizes for the main splitter
            main_splitter.setSizes([1, 6, 3])  # 10% for header, 60% for body, 30% for end

            # Add the main splitter to the main layout
            self.Column_2_Body_model_layout.addWidget(main_splitter)
            
        except Exception as e:
            print(f"Error initializing Column 2 body model layout: {e}")
    
            
    def set_layout_to_train(self):
        page_widget = QWidget()
        layout = QHBoxLayout(page_widget)

        # Left column
        left_column_layout = QVBoxLayout()
        
        button_style = """
            background-color: #416096;
            color: #F9FFFF;
            font-family: Arial;
            font-size: 12;
        """
        
        label_style = """
            background-color: #FCFCFC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        
        list_style = """
            background-color: #FCFCFC;
            color: #000000;
            border: 1px solid #CCCCCC;
        """
           
        combobox_style = """
            color: #F9FFFF;
            font-size: 12px;
            font-family: Arial;
            background-color: #416096;
            border: 1px solid #CCCCCC;
        """     
        lineedit_style = """
            color: #000000;
            font-size: 14px;
            font-family: Arial;
            background-color: #FFFFFF;
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            padding: 5px;
            selection-background-color: #BFE6FF;
            selection-color: #000000;
        """
        
        Table_style = """
        color: #000000;
        font-size: 9px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 2px;
        padding: 2px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """
        
        self.function_create_label("NUMERAI Available Datasets", left_column_layout,label_style)

        self.list_widget_all_datasets = QListWidget()
        self.list_widget_all_datasets.setStyleSheet(list_style)
        self.list_widget_all_datasets.setSelectionMode(QAbstractItemView.MultiSelection)
        left_column_layout.addWidget(self.list_widget_all_datasets)

        # Set a default folder path based on the operating system
        self.folder_path_edit = QLineEdit(self.default_folder_path, self)
        self.folder_path_edit.setStyleSheet(lineedit_style)
        left_column_layout.addWidget(self.folder_path_edit)
        
        self.browse_button = self.function_create_button("Browse", left_column_layout,self.browse_folder, button_style)

        self.button_download_data_set_selected = self.function_create_button("Download Selected Datasets", left_column_layout, self.function_button_download_selected_dataset,button_style)
        
        self.function_create_label("Datasets Downloadeds", left_column_layout,label_style)

        self.list_Widget_Availabile_Downloaded_Datasets = QListWidget()
        self.list_Widget_Availabile_Downloaded_Datasets.setStyleSheet(list_style)
        
        
        self.function_update_downloaded_datasets_list(self.default_folder_path) 
        left_column_layout.addWidget(self.list_Widget_Availabile_Downloaded_Datasets)

        self.function_create_label("Features List - Select One", left_column_layout,label_style)
        
        self.list_Widget_Features_Downloaded_Datasets= QListWidget()
        self.list_Widget_Features_Downloaded_Datasets.setStyleSheet(list_style)
        left_column_layout.addWidget(self.list_Widget_Features_Downloaded_Datasets)
        self.list_Widget_Features_Downloaded_Datasets.itemClicked.connect(self.function_display_feature_list)  

        self.function_create_label("Features List Contents - Select One", left_column_layout,label_style)
        
        self.list_widget_features_content= QListWidget()
        self.list_widget_features_content.setStyleSheet(list_style)
        left_column_layout.addWidget(self.list_widget_features_content)
        self.list_widget_features_content.itemClicked.connect(self.function_handle_feature_list_change)

        self.metadata_widget = QWidget()  
        metadata_layout = QVBoxLayout()    
        self.metadata_widget.setLayout(metadata_layout)
        left_column_layout.addWidget(self.metadata_widget)
        
        layout.addLayout(left_column_layout)
        
        
        # Right column
        right_column_layout = QVBoxLayout()   
        # Section for Training Dataset List       
        self.function_create_label("Training Dataset List:", right_column_layout,label_style)
        
        self.list_widget_train_downloaded_datasets = QListWidget()
        self.list_widget_train_downloaded_datasets.setStyleSheet(list_style)
        right_column_layout.addWidget(self.list_widget_train_downloaded_datasets)
        
        # List of available training methods Q ComboBox
        self.function_create_label("Select Training Module:", right_column_layout,label_style)
        self.training_methods = ["LGBMRegressor", "HistGradientBoostingRegressor"]  
        self.training_method_combo = QComboBox()
        self.training_method_combo.setStyleSheet(combobox_style)
        self.training_method_combo.addItems(self.training_methods)
        right_column_layout.addWidget(self.training_method_combo)

        # Connect signal for combobox selection change
        self.training_method_combo.currentIndexChanged.connect(self.function_training_method_changed)
        
        # Create layout for LGBMRegressor hyperparameters
        self.hyperparameters_layout_LGM = QVBoxLayout()
        widgets_LGM = []  # List to store widgets for LGBMRegressor parameters
        
        self.function_create_label("Number of boosted trees to fit", self.hyperparameters_layout_LGM,label_style)
        self.n_estimators = QLineEdit()
        self.n_estimators.setStyleSheet(lineedit_style)
        self.n_estimators.setPlaceholderText("2000")
        self.hyperparameters_layout_LGM.addWidget(self.n_estimators)
        widgets_LGM.append(self.n_estimators)
        
        self.function_create_label("Boosting learning rate", self.hyperparameters_layout_LGM,label_style)
        self.learning_rate = QLineEdit()
        self.learning_rate.setStyleSheet(lineedit_style)
        self.learning_rate.setPlaceholderText("0.01")
        self.hyperparameters_layout_LGM.addWidget(self.learning_rate)
        widgets_LGM.append(self.learning_rate)
        
        self.function_create_label("Maximum tree depth for base learners", self.hyperparameters_layout_LGM,label_style)
        self.max_depth = QLineEdit()
        self.max_depth.setStyleSheet(lineedit_style)
        self.max_depth.setPlaceholderText("5")
        self.hyperparameters_layout_LGM.addWidget(self.max_depth)
        widgets_LGM.append(self.max_depth)
        
        self.function_create_label("Maximum tree leaves for base learners", self.hyperparameters_layout_LGM,label_style)
        self.num_leaves = QLineEdit()
        self.num_leaves.setStyleSheet(lineedit_style)
        self.num_leaves.setPlaceholderText("5")
        self.hyperparameters_layout_LGM.addWidget(self.num_leaves)
        widgets_LGM.append(self.num_leaves)
        
        self.function_create_label("Subsample ratio of columns when constructing each tree", self.hyperparameters_layout_LGM,label_style)
        self.colsample_bytree = QLineEdit()
        self.colsample_bytree.setStyleSheet(lineedit_style)
        self.colsample_bytree.setPlaceholderText("0.1")
        self.hyperparameters_layout_LGM.addWidget(self.colsample_bytree)
        widgets_LGM.append(self.colsample_bytree)
        
        
        right_column_layout.addLayout(self.hyperparameters_layout_LGM)

        # Create layout for Scitkit hyperparameters
        self.hyperparameters_layout_HistGradientBoostingRegressor = QVBoxLayout()
        
        widgets_HGBR =[]   # Dictionary to store widgets for Scikit regressor parameters

        self.function_create_label("Learning Rate", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        self.learning_rate_sci = QLineEdit()
        self.learning_rate_sci.setStyleSheet(lineedit_style)
        self.learning_rate_sci.setPlaceholderText("0.1")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.learning_rate_sci)
        widgets_HGBR.append(self.learning_rate_sci)
        
        self.function_create_label("Max Number of Trees", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        self.max_iter_sci = QLineEdit()
        self.max_iter_sci.setStyleSheet(lineedit_style)    
        self.max_iter_sci.setPlaceholderText("100")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_iter_sci)
        widgets_HGBR.append(self.max_iter_sci )        
        
        self.function_create_label("Max Num of leaves per Tree", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        self.max_leaf_nodes_sci = QLineEdit()
        self.max_leaf_nodes_sci.setStyleSheet(lineedit_style)  
        self.max_leaf_nodes_sci.setPlaceholderText("31")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_leaf_nodes_sci)
        widgets_HGBR.append(self.max_leaf_nodes_sci)    
        
        self.function_create_label("Max Depth of Tree", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        self.max_depth_sci = QLineEdit()
        self.max_depth_sci.setStyleSheet(lineedit_style)  
        self.max_depth_sci.setPlaceholderText("None")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_depth_sci)
        widgets_HGBR.append(self.max_depth_sci)    
        
        self.function_create_label("Proportion of random chosen features in each node split", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        self.max_features_sci = QLineEdit()
        self.max_features_sci.setStyleSheet(lineedit_style)  
        self.max_features_sci.setPlaceholderText("0.1")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_features_sci)
        widgets_HGBR.append(self.max_features_sci)    
        
        # Initially hide the layout for "Other" hyperparameters
        for i in range(self.hyperparameters_layout_HistGradientBoostingRegressor.count()):
            widget = self.hyperparameters_layout_HistGradientBoostingRegressor.itemAt(i).widget()
            if widget:
                widget.hide()
        
        right_column_layout.addLayout(self.hyperparameters_layout_HistGradientBoostingRegressor)

        # Create download button
        self.button_download_data_set_selected = self.function_create_button("Train", right_column_layout, self.function_Button_Train_Model, button_style)
        self.button_download_data_set_selected = self.function_create_button("Train Multi Models", right_column_layout, self.function_Multiple_Train_Buttons, button_style)

        
        
        
        self.function_create_label("First 50 Rows Training", right_column_layout,label_style)
        self.table_widget_train_dataset = QTableWidget()
        self.table_widget_train_dataset .setStyleSheet(Table_style)
        right_column_layout.addWidget(self.table_widget_train_dataset)
                
        
        layout.addLayout(right_column_layout)
            
        return page_widget
    
    

    def set_layout_to_validate(self):
        
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)

        button_style = """
                background-color: #416096;
                color: #F9FFFF;
                font-family: Arial;
                font-size: 12;
            """
        
        label_style = """
            background-color: #FCFCFC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        
        list_style = """
            background-color: #FCFCFC;
            color: #000000;
            border: 1px solid #CCCCCC;
        """
               
        Table_style = """
        color: #000000;
        font-size: 9px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 2px;
        padding: 2px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """
        # Section for Validation Dataset List      
        self.function_create_label("Validation list:", layout,label_style)
        
        self.list_widget_validation_downloaded_datasets= QListWidget()     
        self.list_widget_validation_downloaded_datasets.setStyleSheet(list_style)   
        layout.addWidget(self.list_widget_validation_downloaded_datasets)
        
        # Create Validate button
        self.button_validate_data_Set_selected = self.function_create_button("Validate", layout, self.function_Button_Validate_Model,button_style)
      
        self.function_create_label("First 50 Rows Validation Table", layout,label_style)
        
        self.table_widget_validation_dataset = QTableWidget()
        self.table_widget_validation_dataset.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_validation_dataset)
        
        self.function_create_label("Validation Results Table", layout,label_style)
        
        self.table_widget_validation_results = QTableWidget()
        self.table_widget_validation_results.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_validation_results)
        
        return page_widget


    def set_layout_to_peformance(self):
        button_style = """
                background-color: #416096;
                color: #F9FFFF;
                font-family: Arial;
                font-size: 12;
            """
        
        label_style = """
            background-color: #FCFCFC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        
        list_style = """
            background-color: #FCFCFC;
            color: #000000;
            border: 1px solid #CCCCCC;
        """
           
        combobox_style = """
            color: #F9FFFF;
            font-size: 12px;
            font-family: Arial;
            background-color: #416096;
            border: 1px solid #CCCCCC;
        """     

        Table_style = """
        color: #000000;
        font-size: 9px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 2px;
        padding: 2px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """        
        
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)

        
        # Section for Validation list
        self.function_create_label("Performance metrics List:", layout,label_style)
        
        self.list_Widget_meta_model_datasets = QListWidget()
        self.list_Widget_meta_model_datasets.setStyleSheet(list_style)
        layout.addWidget(self.list_Widget_meta_model_datasets)
        
        # Create Peformance Evaluation button
        self.button_performance_evaluation_data_set_selected = self.function_create_button("Performance Evaluation", layout, self.function_Button_Evaluate_Peforance_Model,button_style)
        
        #Table for the meta model perforance metricx
        self.function_create_label("Meta Model Performance Table", layout,label_style)
        
        self.table_widget_metamodel_performance_file = QTableWidget()
        self.table_widget_metamodel_performance_file.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_metamodel_performance_file)
        
        self.function_create_label("Validation Results Table", layout,label_style)
        
        self.table_widget_performance_results = QTableWidget()
        self.table_widget_performance_results.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_performance_results)
          
        # Section for Live Prediction List
        self.function_create_label("Live Prediction List:", layout,label_style)
        
        self.list_Widget_Live_Prediction_Datasets = QListWidget()
        self.list_Widget_Live_Prediction_Datasets.setStyleSheet(list_style)
        layout.addWidget(self.list_Widget_Live_Prediction_Datasets)        
        
        # Create Peformance Evaluation button
        self.button_Live_Prediction_Data_Set_Selected = self.function_create_button("Live Prediction:", layout, self.function_Button_Live_predictions_Model,button_style)      
                
        return page_widget
    
    
    
    def set_layout_to_Automate(self):
        button_style = """
                background-color: #416096;
                color: #F9FFFF;
                font-family: Arial;
                font-size: 12;
            """
        label_style = """
            background-color: #FCFCFC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        list_style = """
            background-color: #FCFCFC;
            color: #000000;
            border: 1px solid #CCCCCC;
        """
        combobox_style = """
            color: #F9FFFF;
            font-size: 12px;
            font-family: Arial;
            background-color: #416096;
            border: 1px solid #CCCCCC;
        """     
        Table_style = """
        color: #000000;
        font-size: 9px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 2px;
        padding: 2px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """                
        lineedit_style = """
        color: #000000;
        font-size: 14px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 5px;
        padding: 5px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        self.table_widget_multi_results = QTableWidget()
        self.table_widget_multi_results.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_multi_results)
        
        return page_widget
  
    def hide_other_layouts(self):
        """
        Hide all the layouts related to different steps except the one associated with the clicked button.
        """
        sender_button = self.sender()
        sender_name = sender_button.text()
        
        # Determine which layout widgets to show based on the clicked button's name
        if sender_name == "Train":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(0)
                    
        elif sender_name == "Validate":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(1)
            
        elif sender_name == "Peformance":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(2)

        elif sender_name == "Muti-Automate":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(3)
                
    def function_create_label(self, text: str, parent_layout, style = None)-> None:
        label = QLabel(text)
        label.setFixedHeight(20)
        if style:
            label.setStyleSheet(style)
            
        parent_layout.addWidget(label)
        
    def function_create_button(self, text: str, parent_layout, clicked_handler=None, style = None) -> None:
        button = QPushButton(text)
        button.setFixedHeight(30) 
        if style:
            button.setStyleSheet(style)
        if clicked_handler:
            button.clicked.connect(clicked_handler)
             
        parent_layout.addWidget(button)
         
    def function_handle_feature_list_change(self, selected_item):
        
        if selected_item:
            self.selected_feature_set = selected_item.text()

            # Select Feature File to get the data of the feature set selected
            self.selected_feature_file = self.list_Widget_Features_Downloaded_Datasets.currentItem().text()

            file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
            full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)
            
            self.num_of_features = 0  # Default value in case of an error

            try:
                if file_extension_feature_file == ".json":
                    with open(full_file_path_feature_file, 'r') as file:
                        feature_metadata = json.load(file)
                        selected_feature_sets = feature_metadata["feature_sets"][self.selected_feature_set]
                        self.num_of_features = len(selected_feature_sets)
            except Exception as e:
                return f"Error loading dataset: {e}"
            

        # Clear the existing layout in the metadata widget
        layout = self.metadata_widget.layout()
        if layout:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
        
        label_style = """
            background-color: #DEDEDC;
            color: #000000;
            font-family: Arial;
            font-size: 12px;
        """
        
        if self.selected_feature_file:
            self.function_create_label(f"Feature List File: {self.selected_feature_file}",layout,label_style)
        else:
            QLabel("Feature List File: N/A")

        if self.selected_feature_set:
            self.function_create_label(f"Feature List Selected: {self.selected_feature_set}",layout,label_style)
        else:
            QLabel("Feature List Selected: N/A")
        
        if self.num_of_features:
            self.function_create_label(f"Number of Features: {self.num_of_features}",layout,label_style)
        else:
            QLabel("Number of Features: N/A")
                
    def get_default_folder_path(self):
        system = platform.system()
        if system == "Windows":
            # Default folder path in Windows
            default_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        elif system == "Darwin":  # macOS
            # Default folder path in macOS
            default_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        elif system == "Linux":
            # Default folder path in Linux
            default_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        else:
            # Default to user's home directory
            default_folder = os.path.expanduser("~")

        # Create a sandbox folder within the default Downloads folder
        sandbox_folder = os.path.join(default_folder, "sandbox")

        # Check if the sandbox folder exists
        if not os.path.exists(sandbox_folder):
            try:
                # Create the sandbox folder if it doesn't exist
                os.makedirs(sandbox_folder)
            except Exception as e:
                print(f"Failed to create sandbox folder: {e}")
        
        self.dynamic_folder_path = sandbox_folder
        
        return sandbox_folder

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)    
            self.function_update_downloaded_datasets_list(folder_path)          
            self.dynamic_folder_path = folder_path
        
    def function_training_method_changed(self):
        selected_method = self.training_method_combo.currentText()
        if selected_method == "HistGradientBoostingRegressor":
            # Hide widgets within the layout for LGBMRegressor hyperparameters and show widgets within the layout for "Other" hyperparameters
            self.function_hide_widgets_in_layout_single(self.hyperparameters_layout_LGM)
            self.function_show_widgets_in_layout_single(self.hyperparameters_layout_HistGradientBoostingRegressor)
        elif selected_method == "LGBMRegressor":
            # Show widgets within the layout for LGBMRegressor hyperparameters and hide widgets within the layout for "Other" hyperparameters
            self.function_show_widgets_in_layout_single(self.hyperparameters_layout_LGM)
            self.function_hide_widgets_in_layout_single(self.hyperparameters_layout_HistGradientBoostingRegressor)
        
    def function_hide_widgets_in_layout_single(self, layout):
        """
        Recursively hide all widgets within the given layout and its nested layouts.
        """
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().hide()

    def function_show_widgets_in_layout_single(self, layout):
        """
        Show all widgets within the given layout and hide all widgets in other layouts.
        """
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().show()
                   
    def function_Download_all_available_datasets_numerAi(self)-> None:
        # Clear existing items from the list widget
        self.list_widget_all_datasets.clear()
        # Initialize NumerAPI - the official Python API client for Numerai
        napi = NumerAPI()
        # list the datasets and available versions
        try:
            all_datasets = napi.list_datasets()
        except Exception as e:
            print(f"Error retrieving datasets from Numerai: {e}")
        #add to list widget     
        self.list_widget_all_datasets.addItems(all_datasets)
                                        
    def function_update_downloaded_datasets_table(self, dataset=None) -> None:
        """
        Updates the list of downloaded datasets.
        """
        try:       

            # Check if each list widget is defined before attempting to collect existing items
            if hasattr(self, 'list_widget_train_downloaded_datasets'):
                existing_items_train = {self.list_widget_train_downloaded_datasets.item(i).text()
                                        for i in range(self.list_widget_train_downloaded_datasets.count())}
    

            if hasattr(self, 'list_widget_validation_downloaded_datasets'):
                existing_items_validation = {self.list_widget_validation_downloaded_datasets.item(i).text()
                                            for i in range(self.list_widget_validation_downloaded_datasets.count())}
        

            if hasattr(self, 'list_Widget_Features_Downloaded_Datasets'):
                existing_items_Features = {self.list_Widget_Features_Downloaded_Datasets.item(i).text()
                                        for i in range(self.list_Widget_Features_Downloaded_Datasets.count())}
           
            if hasattr(self, 'list_Widget_Availabile_Downloaded_Datasets'):
                existing_items_downloaded = {self.list_Widget_Availabile_Downloaded_Datasets.item(i).text()
                                            for i in range(self.list_Widget_Availabile_Downloaded_Datasets.count())}
        

            if hasattr(self, 'list_Widget_meta_model_datasets'):
                existing_Meta_Model_downloaded = {self.list_Widget_meta_model_datasets.item(i).text()
                                                for i in range(self.list_Widget_meta_model_datasets.count())}
        

            if hasattr(self, 'list_Widget_Live_Prediction_Datasets'):
                existing_Live_Prediction_downloaded = {self.list_Widget_Live_Prediction_Datasets.item(i).text()
                                                    for i in range(self.list_Widget_Live_Prediction_Datasets.count())}
                  
        except AttributeError as e:
            print("List widgets are not yet defined. Skipping function execution.")
            return
      
        # Iterate through each item in the list of downloaded datasets
        # Convert the item to lowercase for case-insensitive comparison
        # Check if the item contains a word and is not already in the list for that word
        for item in existing_items_downloaded:
            item_lower = item.lower()
            if "feature" in item_lower and item not in existing_items_Features:
                self.list_Widget_Features_Downloaded_Datasets.addItem(item)

            if "validation" in item_lower and item not in existing_items_validation:
                self.list_widget_validation_downloaded_datasets.addItem(item)

            if "train" in item_lower and item not in existing_items_train:
                self.list_widget_train_downloaded_datasets.addItem(item)

            if "meta_model" in item_lower and item not in existing_Meta_Model_downloaded:
                self.list_Widget_meta_model_datasets.addItem(item)

            if "live" in item_lower and item not in existing_Live_Prediction_downloaded:
                self.list_Widget_Live_Prediction_Datasets.addItem(item)

        # Dataset is triggered when a user clicks download button. 
        # Extract the text from the dataset object selected for download.
        # Add the dataset to the available downloaded datasets widget if it's not already there
        # Check if the dataset contains word  and is not already in the list of existing datasets
        if dataset is not None:    
            dataset= dataset.text()
            if dataset not in existing_items_downloaded:
                self.list_Widget_Availabile_Downloaded_Datasets.addItem(dataset)

            if "feature" in dataset.lower() and dataset not in existing_items_Features:
                self.list_Widget_Features_Downloaded_Datasets.addItem(dataset)
                
            if "validation" in dataset.lower() and dataset not in existing_items_validation:
                self.list_widget_validation_downloaded_datasets.addItem(dataset)
                
            if "train" in dataset.lower() and dataset not in existing_items_train:
                self.list_widget_train_downloaded_datasets.addItem(dataset)
        
            if "meta_model" in dataset.lower() and item not in existing_Meta_Model_downloaded:
                self.list_Widget_meta_model_datasets.addItem(dataset)
                
            if "live" in dataset.lower() and item not in existing_Live_Prediction_downloaded:
                self.list_Widget_Live_Prediction_Datasets.addItem(dataset)
                
    def function_update_downloaded_datasets_list(self,folder_path) -> None:
        """
        Updates the list of downloaded datasets.
        """
        self.list_Widget_Availabile_Downloaded_Datasets.clear()
        try:
        # Add downloaded datasets to the table
            for root, dirs, files in os.walk(folder_path):
                for dir in dirs:
                    # Get the full directory path
                    dir_path = os.path.join(root, dir)
                    # Iterate over files in the directory
                    for file in os.listdir(dir_path):
                        # Check if file has one of the desired extensions
                        if file.endswith(('.csv', '.parquet', '.json', '.xlsx', '.db', '.sqlite', '.sqlite3')):
                            # Get the full file path
                            file_path = os.path.join(dir, file)
                            file_path = file_path.replace('\\', '/')
                            # Add dataset to the list if it doesn't already exist
                            self.list_Widget_Availabile_Downloaded_Datasets.addItem(file_path)
            self.function_update_downloaded_datasets_table()
        except Exception as e:
            print(f"Error updating downloaded datasets list: {e}")
                                      
    def function_button_download_selected_dataset(self)->None:
        # Initialize NumerAPI - the official Python API client for Numerai
        napi = NumerAPI()
        List_selected_Datasets = self.list_widget_all_datasets.selectedItems()
        
        for dataset in List_selected_Datasets:
            dataset_text = dataset.text()  # Get the text of the selected dataset
            print(dataset_text)
            # Download dataset and save it in the sandbox directory
            if self.dynamic_folder_path:
                file_path = os.path.join(self.dynamic_folder_path, f"{dataset.text()}")
                napi.download_dataset(dataset.text(), dest_path=file_path)
                # Update the downloaded datasets table
                self.function_update_downloaded_datasets_table(dataset)  
                     
    def function_display_feature_list(self, item) ->None: 
        self.selected_feature_file = item.text()
        file_extension = os.path.splitext(self.selected_feature_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

        try:
            if file_extension == ".json":
                with open(full_file_path, 'r') as file:
                    feature_metadata = json.load(file)
                # Clear existing contents of the list widget
                self.list_widget_features_content.clear()
                # Get feature sets data
                feature_sets_data = feature_metadata.get('feature_sets', {})
                # Populate list widget with feature sets keys
                for key in feature_sets_data.keys():
                    item = QListWidgetItem(str(key))
                    self.list_widget_features_content.addItem(item)
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return

    def fuction_parquet_data_into_table(self, df, table_widget)->None:
        # Clear existing contents of the table widget
        self.clear_table_widget(table_widget)
        
        # Limit the maximum number of rows to display
        max_rows = 50  # Adjust as needed
        df_head = df.head(max_rows)
        
        # Set the column count and headers
        table_widget.setColumnCount(len(df_head.columns) + 1)
        headers = ["ID"] + list(df_head.columns)
        table_widget.setHorizontalHeaderLabels(headers)
        
            
        for row_num, (index, row) in enumerate(df_head.iterrows()):
            table_widget.insertRow(row_num)
            table_widget.setItem(row_num, 0, QTableWidgetItem(str(index)))

            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                table_widget.setItem(row_num, j+1, item)
                          
        # Get the number of rows and columns in the DataFrame
        num_rows, num_cols = df.shape
        
        # Print the number of rows and columns
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_cols}")
        
           
    def clear_table_widget(self, table_widget)->None:
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)
    
    
    def function_Button_Train_Model(self) -> None:
        
        # Clear trained file and model global parameters
        self.read_parquet_train_file = None
        self.trained_model = None
        self.selected_feature_sets_all_features = None
        # Verify modelling method algorithm selected by user 
        selected_method = self.training_method_combo.currentText()
        
        if selected_method == "HistGradientBoostingRegressor":
            # Get hyperparameters values from GUI inputs
            learning_rate_value = self.learning_rate_sci.text()
            n_estimators_value=self.max_iter_sci.text()
            num_leaves_value=self.max_leaf_nodes_sci.text()
            max_depth_value= self.max_depth_sci.text()
            colsample_bytree_value = self.max_features_sci.text()
 
            # Validate hyperparameters
            if not all([n_estimators_value, learning_rate_value, max_depth_value, num_leaves_value, colsample_bytree_value]):
                QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
                return None
            
            if not self.selected_feature_set or not self.selected_feature_file:
                QMessageBox.warning(self, "Selection Required", "Please select a Feature Set and a Feature File to proceed.")
                return
        
            # Convert hyperparameters to appropriate types
            try:
                hyperparameters = {
                    'max_iter': int(n_estimators_value),
                    'learning_rate': float(learning_rate_value),
                    'max_depth': int(max_depth_value),
                    'max_leaf_nodes': int(eval(num_leaves_value)),
                    'max_features': float(colsample_bytree_value)
                }
                
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid hyperparameter values.")
                return None
    
            file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
            full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

            try:
                if file_extension_feature_file == ".json":
                    with open(full_file_path_feature_file, 'r') as file:
                        feature_metadata = json.load(file)
                        self.selected_feature_sets_all_features = feature_metadata["feature_sets"][self.selected_feature_set]
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return

            # Define feature set and download training data
            selected_training_file =  self.list_widget_train_downloaded_datasets.currentItem()
            if not selected_training_file:  # No item is selected
                QMessageBox.warning(self, "Selection Required", "Please select a training int 8 file to proceed.")
                return
            
            selected_training_file = selected_training_file.text()
            file_extension_training_file = os.path.splitext(selected_training_file)[1].lower()
            
            try:
                if file_extension_training_file == ".parquet":
                    self.read_parquet_train_file= pd.read_parquet(selected_training_file, columns=["era"] + self.selected_feature_sets_all_features +["target"])
                    self.fuction_parquet_data_into_table(self.read_parquet_train_file,self.table_widget_train_dataset)
                else:
                    raise ValueError("Unsupported file format.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return
            
            # Train the model using the provided features and target data
            self.trained_model = HistGradientBoostingRegressor(**hyperparameters)
            self.trained_model.fit(self.read_parquet_train_file[self.selected_feature_sets_all_features],self.read_parquet_train_file["target"])
            
            print("Model Training Ended")
            
            
        elif selected_method == "LGBMRegressor":
            # Get hyperparameters values from GUI inputs
            n_estimators_value = self.n_estimators.text()
            learning_rate_value= self.learning_rate.text()
            max_depth_value= self.max_depth.text()
            num_leaves_value= self.num_leaves.text()
            colsample_bytree_value =self.colsample_bytree.text()
            
            # Validate hyperparameters
            if not all([n_estimators_value, learning_rate_value, max_depth_value, num_leaves_value, colsample_bytree_value]):
                QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
                return None
            
            if not self.selected_feature_set or not self.selected_feature_file:
                QMessageBox.warning(self, "Selection Required", "Please select a Feature Set and a Feature File to proceed.")
                return
          
            # Convert hyperparameters to appropriate types
            try:
                hyperparameters = {
                    'n_estimators': int(n_estimators_value),
                    'learning_rate': float(learning_rate_value),
                    'max_depth': int(max_depth_value),
                    'num_leaves': int(eval(num_leaves_value)),
                    'colsample_bytree': float(colsample_bytree_value)
                }

            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid hyperparameter values.")
                return None
    
            file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
            full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

            try:
                if file_extension_feature_file == ".json":
                    with open(full_file_path_feature_file, 'r') as file:
                        feature_metadata = json.load(file)
                        self.selected_feature_sets_all_features = feature_metadata["feature_sets"][self.selected_feature_set]
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return

            # Define feature set and download training data
            selected_training_file =  self.list_widget_train_downloaded_datasets.currentItem()
            
            if not selected_training_file:  # No item is selected
                QMessageBox.warning(self, "Selection Required", "Please select a training int 8 file to proceed.")
                return
            
            selected_training_file = selected_training_file.text()
            file_extension_training_file = os.path.splitext(selected_training_file)[1].lower()
            full_file_path_training_file = os.path.join(self.dynamic_folder_path, selected_training_file)

            try:
                if file_extension_training_file == ".parquet":
                    with open(full_file_path_training_file, 'r') as file:
                        self.read_parquet_train_file = pd.read_parquet(full_file_path_training_file, columns=["era"] + self.selected_feature_sets_all_features +["target"])
                        self.fuction_parquet_data_into_table(self.read_parquet_train_file,self.table_widget_train_dataset)
                else:
                    raise ValueError("Unsupported file format.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return
            # Train the model using the provided features and target data
            self.trained_model = LGBMRegressor(**hyperparameters)
            self.trained_model.fit(self.read_parquet_train_file[self.selected_feature_sets_all_features],self.read_parquet_train_file["target"])
            print("Model Training for LGB Regressor Ended")




    def function_Button_Validate_Model(self) -> None:
        
        """
        Validates the trained model using the validation dataset.
        """
        if not self.selected_feature_set:  # No item is selected
            QMessageBox.warning(self, "Selection Required", "Please select an Feature Set to proceed.")
            return       
        
        if not self.selected_feature_file:  # No item is selected
            QMessageBox.warning(self, "Selection Required", "Please select a featue file to proceed.")
            return
        
        file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
        full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

        try:
            if file_extension_feature_file == ".json":
                with open(full_file_path_feature_file, 'r') as file:
                    feature_metadata = json.load(file)
                    selected_feature_sets = feature_metadata["feature_sets"][self.selected_feature_set]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return
           
        #Select Validation File 
        Selected_Validation_File = self.list_widget_validation_downloaded_datasets.currentItem()
        if not Selected_Validation_File:  # No item is selected
            QMessageBox.warning(self, "Selection Required", "Please select a Validation int 8 file to proceed.")
            return
        
        Selected_Validation_File = Selected_Validation_File.text()
        file_extension_Validation_file = os.path.splitext(Selected_Validation_File)[1].lower()
        
        full_file_path_validation_file = os.path.join(self.dynamic_folder_path, Selected_Validation_File)


        try:
            if file_extension_Validation_file == ".parquet":
                validation = pd.read_parquet(full_file_path_validation_file, columns=["era" ,"data_type", "target"] + selected_feature_sets)
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return
        
        validation = validation[validation["data_type"] == "validation"]
        del validation["data_type"]

        # Eras are 1 week apart, but targets look 20 days (or 4 weeks/eras) into the future,
        # so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
        last_train_era = int(self.read_parquet_train_file["era"].unique()[-1])
        eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
        validation = validation[~validation["era"].isin(eras_to_embargo)]
        
        self.fuction_parquet_data_into_table(validation,self.table_widget_validation_dataset)

        # Generate predictions against the out-of-sample validation features
        # This will take a few minutes
        validation["prediction"] = self.trained_model.predict(validation[selected_feature_sets])
        print(validation[["era", "prediction", "target"]])
        
        # Clear existing contents of the table widget
        self.clear_table_widget(self.table_widget_validation_results)
        
        # Set the column count and headers
        num_columns = 3  # era, prediction, target
        self.table_widget_validation_results.setColumnCount(num_columns + 1)  # Add 1 for the index column
        headers = ["Index", "Era", "Prediction", "Target"]
        self.table_widget_validation_results.setHorizontalHeaderLabels(headers)
        
        # Extract first and last 50 rows with index
        validation_subset = pd.concat([validation.head(50), validation.tail(50)])
        
        # Populate the table with validation results
        for row_num, (index, row) in enumerate(validation_subset.iterrows()):
            self.table_widget_validation_results.insertRow(row_num)
            self.table_widget_validation_results.setItem(row_num, 0, QTableWidgetItem(str(index)))  # Index value
            self.table_widget_validation_results.setItem(row_num, 1, QTableWidgetItem(str(row["era"])))
            self.table_widget_validation_results.setItem(row_num, 2, QTableWidgetItem(str(row["prediction"])))
            self.table_widget_validation_results.setItem(row_num, 3, QTableWidgetItem(str(row["target"])))
        
        self.Validation_Model = validation
        
    def function_Button_Evaluate_Peforance_Model(self) -> None:
        """
        Performance Evaluation
        """
        #Select Peformance Model File 
        Selected_Performance_Model_File = self.list_Widget_meta_model_datasets.currentItem()
        if not Selected_Performance_Model_File:  # No item is selected
            QMessageBox.warning(self, "Selection Required", "Please select a Validation int 8 file to proceed.")
            return
        
        Selected_Performance_Model_File = Selected_Performance_Model_File.text()
        file_extension_Training_file = os.path.splitext(Selected_Performance_Model_File)[1].lower()
        
        full_file_path_peformance_file = os.path.join(self.dynamic_folder_path, Selected_Performance_Model_File)

        try:
            if file_extension_Training_file == ".parquet":
                self.Validation_Model["meta_model"]= pd.read_parquet(full_file_path_peformance_file)["numerai_meta_model"]
                self.fuction_parquet_data_into_table(self.Validation_Model,self.table_widget_metamodel_performance_file)

            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return
    
        # # Compute the per-era corr between our predictions and the target values
        # per_era_corr = self.Validation_Model.groupby("era").apply(
        #     lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna()),
        #      include_groups=False
        # )

        
        # per_era_mmc = self.Validation_Model.dropna().groupby("era").apply(
        #     lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]),
        #     include_groups=False
        # )

        # Compute the per-era corr between our predictions and the target values
        per_era_corr = self.Validation_Model.groupby("era").apply(
            lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
            )

        
        per_era_mmc = self.Validation_Model.dropna().groupby("era").apply(
            lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]) 
            )


        # Compute performance metrics
        corr_mean = per_era_corr.mean()
        corr_std = per_era_corr.std(ddof=0)
        corr_sharpe = corr_mean / corr_std
        corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

        mmc_mean = per_era_mmc.mean()
        mmc_std = per_era_mmc.std(ddof=0)
        mmc_sharpe = mmc_mean / mmc_std
        mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

        # Create a DataFrame to hold the results
        results_df = pd.DataFrame({
            "mean": [corr_mean, mmc_mean],
            "std": [corr_std, mmc_std],
            "sharpe": [corr_sharpe, mmc_sharpe],
            "max_drawdown": [corr_max_drawdown, mmc_max_drawdown]
        }, index=["CORR", "MMC"]).T
            
        # Print the results DataFrame
        print("Performance Metrics:")
        print(results_df)    
           
        self.fuction_parquet_data_into_table(results_df,self.table_widget_performance_results)


    def function_Button_Live_predictions_Model(self) -> None:
        self.live_features_stored = None

        Selected_Live_Model_File = self.list_Widget_Live_Prediction_Datasets.currentItem()
        if not Selected_Live_Model_File:  # No item is selected
            QMessageBox.warning(self, "Selection Required", "Please select a live file int 8 file to proceed.")
            return
        
        Selected_Live_Model_File = Selected_Live_Model_File.text()
        file_extension_Live_file = os.path.splitext(Selected_Live_Model_File)[1].lower()
        
        full_file_path_Live_file = os.path.join(self.dynamic_folder_path, Selected_Live_Model_File)
        try:
            if file_extension_Live_file == ".parquet":
                self.live_features_stored = pd.read_parquet(full_file_path_Live_file, columns=self.selected_feature_sets_all_features)
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return

        if self.trained_model is None or self.selected_feature_sets_all_features is None:
            QMessageBox.warning(None, "Error", "Please train the model first.")
            return None

        # Create an instance of ModelWithPredictMethod
        model_with_predict = ModelWithPredictMethod(self.trained_model, self.selected_feature_sets_all_features)

        try:
            # Pickle the entire ModelWithPredictMethod class
            model_pickle = cloudpickle.dumps(model_with_predict.predict)
            with open("model.pkl", "wb") as f:
                f.write(model_pickle)
            QMessageBox.information(None, "Success", "Model with predict method pickled successfully.")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Error pickling model: {e}")
            

        # Generate live predictions
        live_predictions = self.trained_model.predict(self.live_features_stored[self.selected_feature_sets_all_features])
        # Format submission
        live_predictions_dataframe = pd.Series(live_predictions, index=self.live_features_stored.index).to_frame("prediction")
        
        print("Predictions file downloaded")
        # Print the results DataFrame
        print("Live Predictions:")
        print(live_predictions_dataframe)  
       
        # Save the DataFrame to a CSV file
        csv_file_path = "live_predictions.csv"
        live_predictions_dataframe.to_csv(csv_file_path)

        print(f"Live predictions saved to {csv_file_path}")
        
                        
             
    def function_Multiple_Train_Buttons(self) -> None: 
        # Clear trained file and model global parameters
        self.read_parquet_train_file = None
        self.trained_model = None
        self.selected_feature_sets_all_features = None   
        
        # Verify modelling method algorithm selected by user 
        selected_method = self.training_method_combo.currentText()
        
        if selected_method == "LGBMRegressor":
            # Get hyperparameters values from GUI inputs
            n_estimators_value = self.n_estimators.text()
            learning_rate_value = self.learning_rate.text()
            max_depth_value = self.max_depth.text()
            num_leaves_value = self.num_leaves.text()
            colsample_bytree_value = self.colsample_bytree.text()
            
            # Validate hyperparameters
            if not all([n_estimators_value, learning_rate_value, max_depth_value, num_leaves_value, colsample_bytree_value]):
                QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
                return None
            
            # Split user input based on delimiter
            n_estimators_values = re.split(r'\s|[,;]', n_estimators_value)
            learning_rate_values = re.split(r'\s|[,;]', learning_rate_value)
            max_depth_values = re.split(r'\s|[,;]', max_depth_value)
            num_leaves_values = re.split(r'\s|[,;]', num_leaves_value)
            colsample_bytree_values = re.split(r'\s|[,;]', colsample_bytree_value)
            
            # Get the lengths of each parameter list
            lengths = {
                'n_estimators': len(n_estimators_values),
                'learning_rate': len(learning_rate_values),
                'max_depth': len(max_depth_values),
                'num_leaves': len(num_leaves_values),
                'colsample_bytree': len(colsample_bytree_values)
            }

            # Check if all lengths are equal
            if len(set(lengths.values())) != 1:
                print("Number of values for each hyperparameter is not equal.")
                return False

            # Create hyperparameters grid based on user input
            hyperparameters_grid = {
                'n_estimators': [int(value) for value in n_estimators_values],
                'learning_rate': [float(value) for value in learning_rate_values],
                'max_depth': [int(value) for value in max_depth_values],
                'num_leaves': [int(eval(value)) for value in num_leaves_values],
                'colsample_bytree': [float(value) for value in colsample_bytree_values]
            }


            # Validate hyperparameters
            if not all([n_estimators_value, learning_rate_value, max_depth_value, num_leaves_value, colsample_bytree_value]):
                QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
                return None
            
            if not self.selected_feature_set or not self.selected_feature_file:
                QMessageBox.warning(self, "Selection Required", "Please select a Feature Set and a Feature File to proceed.")
                return
            

            # Create an empty DataFrame to store the results
            results_df = pd.DataFrame(columns=['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'colsample_bytree'])

            # Create an empty list to store the results DataFrames
            results_dfs = []
        
            # Clear trained file and model global parameters
            self.read_parquet_train_file = None
            self.trained_model = None
            
            if not self.selected_feature_set:  # No item is selected
                QMessageBox.warning(self, "Selection Required", "Please select an Feature Set to proceed.")
                return

            if not self.selected_feature_file:  # No item is selected
                QMessageBox.warning(self, "Selection Required", "Please select a featue file to proceed.")
                return
            
            file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
            full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

            try:
                if file_extension_feature_file == ".json":
                    with open(full_file_path_feature_file, 'r') as file:
                        feature_metadata = json.load(file)
                        selected_feature_sets = feature_metadata["feature_sets"][self.selected_feature_set]
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return

            # Define feature set and download training data
            selected_training_file =  self.list_widget_train_downloaded_datasets.currentItem()
            if not selected_training_file:  # No item is selected
                QMessageBox.warning(self, "Selection Required", "Please select a training int 8 file to proceed.")
                return
            
            selected_training_file = selected_training_file.text()
            file_extension_training_file = os.path.splitext(selected_training_file)[1].lower()
            full_file_path_training_file = os.path.join(self.dynamic_folder_path, selected_training_file)
            
            try:
                if file_extension_training_file == ".parquet":
                    train = pd.read_parquet(full_file_path_training_file, columns=["era", "target"] + selected_feature_sets)
                    self.fuction_parquet_data_into_table(train,self.table_widget_train_dataset)
                else:
                    raise ValueError("Unsupported file format.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                return        
            
                        # Set the current index to activate body_widget
            self.body_widget.setCurrentIndex(1)
            
            # Ask the user to ensure a validation file type is selected
            reply = QMessageBox.question(self, 'Validation File Selection', 'Please ensure a validation file type is selected before continuing. Continue to select your file?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return None

            # Wait for the user to select a validation file
            while not self.list_widget_validation_downloaded_datasets.currentItem():
                QApplication.processEvents()

            # Get the selected validation file and proceed with the rest of the code
            selected_validation_file = self.list_widget_validation_downloaded_datasets.currentItem().text()
            print("Selected Validation File:", selected_validation_file)

            # Set the current index to activate body_widget
            self.body_widget.setCurrentIndex(2)

            # Ask the user to ensure a validation file type is selected
            reply = QMessageBox.question(self, 'Performance Metric File Selection', 'Please ensure a Metric Performance file type is selected before continuing. Continue to select your file?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return None

            # Wait for the user to select a meta model file
            while not self.list_Widget_meta_model_datasets.currentItem():
                QApplication.processEvents()
                
            # Get the selected validation file and proceed with the rest of the code
            selected_peformance_file = self.list_Widget_meta_model_datasets.currentItem().text()
            print("Selected peformance File:", selected_peformance_file)
        
            # Iterate over the hyperparameters grid
            for i in range(len(hyperparameters_grid['n_estimators'])):
                # Extract hyperparameters for this iteration
                n_estimators = hyperparameters_grid['n_estimators'][i]
                learning_rate = hyperparameters_grid['learning_rate'][i]
                max_depth = hyperparameters_grid['max_depth'][i]
                num_leaves = hyperparameters_grid['num_leaves'][i]
                colsample_bytree = hyperparameters_grid['colsample_bytree'][i]
        
                n_estimators_value = n_estimators
                learning_rate_value= learning_rate
                max_depth_value= max_depth
                num_leaves_value= num_leaves
                colsample_bytree_value =colsample_bytree
            
                # Convert hyperparameters to appropriate types
                try:
                    hyperparameters = {
                        'n_estimators': int(n_estimators_value),
                        'learning_rate': float(learning_rate_value),
                        'max_depth': int(max_depth_value),
                        'num_leaves': int(num_leaves_value),
                        'colsample_bytree': float(colsample_bytree_value)
                    }
                    # Use retrieved values to train the model
                    print("Training model with hyperparameters:", hyperparameters)

                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter valid hyperparameter values.")
                    return None

                # Train the model using the provided features and target data
                model = LGBMRegressor(**hyperparameters)
                model.fit(train[selected_feature_sets],train["target"])
                
                # Save trained model and training data for use in next steps
                self.read_parquet_train_file = train
                self.trained_model = model
                print("Model Training for LGB Regressor Ended")
                print("Cycle")
        


                self.function_Button_Validate_Model()
                
                """
                Performance Evaluation
                """
                #Select Peformance Model File 
                Selected_Performance_Model_File = self.list_Widget_meta_model_datasets.currentItem()
                
                if not Selected_Performance_Model_File:  # No item is selected
                    QMessageBox.warning(self, "Selection Required", "Please select a Validation int 8 file to proceed.")
                    return
                
                Selected_Performance_Model_File = Selected_Performance_Model_File.text()
                file_extension_peformance_file = os.path.splitext(Selected_Performance_Model_File)[1].lower()
                full_file_path_peformance_file = os.path.join(self.dynamic_folder_path, Selected_Performance_Model_File)
                try:
                    if file_extension_peformance_file == ".parquet":
                        self.Validation_Model["meta_model"]= pd.read_parquet(full_file_path_peformance_file)["numerai_meta_model"]
                        self.fuction_parquet_data_into_table(self.Validation_Model,self.table_widget_metamodel_performance_file)

                    else:
                        raise ValueError("Unsupported file format.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
                    return
            
                # # Compute the per-era corr between our predictions and the target values
                # per_era_corr = self.Validation_Model.groupby("era").apply(
                #     lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna()),
                #     include_groups=False
                # )

                
                # per_era_mmc = self.Validation_Model.dropna().groupby("era").apply(
                #     lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]),
                #     include_groups=False
                # )
                
                # Compute the per-era corr between our predictions and the target values
                per_era_corr = self.Validation_Model.groupby("era").apply(
                    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
            
                )

                
                per_era_mmc = self.Validation_Model.dropna().groupby("era").apply(
                    lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
                )
                

                # Compute performance metrics
                corr_mean = per_era_corr.mean()
                corr_std = per_era_corr.std(ddof=0)
                corr_sharpe = corr_mean / corr_std
                corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

                mmc_mean = per_era_mmc.mean()
                mmc_std = per_era_mmc.std(ddof=0)
                mmc_sharpe = mmc_mean / mmc_std
                mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

                # Create a DataFrame to hold the results
                results_df = pd.DataFrame({
                    'n_estimators': [n_estimators],
                    'learning_rate': [learning_rate],
                    'max_depth': [max_depth],
                    'num_leaves': [num_leaves],
                    'colsample_bytree': [colsample_bytree],
                    "corr_mean": [corr_mean],
                    "mmc_mean": [mmc_mean],
                    "corr_std": [corr_std],
                    "mmc_std": [mmc_std],
                    "corr_sharpe": [corr_sharpe],
                    "mmc_sharpe": [mmc_sharpe],
                    "corr_max_drawdown": [corr_max_drawdown],
                    "mmc_max_drawdown": [mmc_max_drawdown]
                })
                                
                # Append the results DataFrame to the list
                results_dfs.append(results_df)

            # Concatenate all results DataFrames into a single DataFrame
            results_df = pd.concat(results_dfs, ignore_index=True)

            # Clear existing contents of the table widget
            self.clear_table_widget(self.table_widget_multi_results)



            # Set the column count and headers
            self.table_widget_multi_results.setColumnCount(len(results_df.columns))
            headers = list(results_df.columns)
            self.table_widget_multi_results.setHorizontalHeaderLabels(headers)

            
            for row_num, (index, row) in enumerate(results_df.iterrows()):
                self.table_widget_multi_results.insertRow(row_num)
                # self.table_widget_multi_results.setItem(row_num, 0, QTableWidgetItem(str(index)))

                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.table_widget_multi_results.setItem(row_num, j, item)
                    

            # Set the current index to activate body_widget
            self.body_widget.setCurrentIndex(3)

            # Save the results to a log file
            results_df.to_csv('hyperparameter_results.csv', index=False)
                                            
           

class ModelWithPredictMethod:
    def __init__(self, trained_model, selected_feature_sets_all_features):
        self.trained_model = trained_model
        self.selected_feature_sets_all_features = selected_feature_sets_all_features

    def predict(self, live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = self.trained_model.predict(live_features[self.selected_feature_sets_all_features])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")
     
              
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
 
    def write(self, message):
        self.text_widget.append(message)  # Use appendPlainText to avoid extra white spaces
        self.text_widget.ensureCursorVisible()  # Ensure cursor is visible
        
    def flush(self):
        pass

class StderrRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        # Apply a custom style for error messages
        self.text_widget.append(f'<span style="color:red;">{message}</span>')
        self.text_widget.ensureCursorVisible()  # Ensure cursor is visible
        print(f"Error: {message}")
        
    def flush(self):
        pass
                          
# Create a class for the main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NumerAi Model Training Application by Ali")
        self.setGeometry(100, 100, 800, 600)
        self.platform = Platform(self)  # Initialize the Platform instance with the MainWindow as parent
        self.setCentralWidget(self.platform)

# Create a class to manage the main application logic
class app_Initializer:
    def __init__(self, app_instance):
        self.app = app_instance  # Assign the provided app_instance

    def start(self):
        # self.setup_styles()
        self.platform = Platform(None)  # Passing None as parent
        self.platform.show()
        sys.exit(self.app.exec())  # Call exec_() on the QApplication instance

    # def setup_styles(self):
    #     button_style = '''
    #         QPushButton {
    #             background-color: #474385;
    #             color: #E5E5ED;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 16px;
    #         }
    #     '''
    #     label_style = '''
    #         QLabel {
    #             background-color: #EEEEF1;
    #             color: #050505;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 12px;
    #         }
    #     '''
    #     list_widget_style = '''
    #         QListWidget {
    #             background-color:#DCDBEA;
    #             color:#616065;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 12px;
    #         }
    #     '''
    #     tab_style = '''
    #         QTabWidget::pane {
    #             border: none;
    #             background-color: #DCDBEA;
    #         }
    #         QTabWidget::tab-bar {
    #             alignment: center;
    #             color: #F6F6F9;
    #         }
    #         QTabWidget::tab {
    #             background-color:#DCDBEA;
    #             color: #050505;
    #             font-family: Helvetica;
    #             font-size: 12px;
    #             padding: 10px;
    #         }
    #         QTabBar::tab {
    #             background-color: #DCDBEA;
    #             color: #050505;
    #             font-family: Helvetica;
    #             font-size: 12px;
    #             padding: 10px;
    #         }
    #         QTabBar::tab:selected {
    #             background-color:#DCDBEA;
    #             color: #050505;
    #         }
    #     '''
    #     textedit_style = '''
    #         QTextEdit {
    #             background-color: #DCDBEA;
    #             color: #15141A;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 12px;
    #             border: none;
    #             padding: 10px;
    #         }
    #     '''
    #     lineedit_tablewidget_style = '''
    #         QLineEdit, QTableWidget {
    #             background-color: #DCDBEA;
    #             color: #616065;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 12px;
    #         }
    #     '''
    #     # Style for QComboBox
    #     combobox_style = '''
    #         QComboBox {
    #             background-color: #DCDBEA;
    #             color: #15141A;
    #             font-family: Noto Sans Tangsa;
    #             font-size: 12px;
    #             border: none;
    #             padding: 5px;
    #         }
    #     '''

    #     # Combine all style definitions into one stylesheet string
    #     stylesheet = f'''
    #         Platform {{
    #             background-color: #EEEEF1;
    #         }}
    #     {button_style}{label_style}{list_widget_style}{tab_style}{textedit_style}{lineedit_tablewidget_style}{combobox_style}
    #     '''

    #     # Apply the stylesheet to the app
    #     self.app.setStyleSheet(stylesheet)



# Your Platform class and other classes go here
if __name__ == "__main__":
    app = QApplication(sys.argv)
    initializer = app_Initializer(app)
    initializer.start()



                                      
    # def initialize_Column_1_Train_Data_Set_Layout(self)-> None:
    #     """
    #     Initializes the layout for Column 1, allowing users to view and download available datasets.
    #     """
    #     # Column 1 : View and download available Datasets from Numerai
    #     column_1_layout_internal = QVBoxLayout()

    #     # Create download button
    #     self.button_download_data_set_selected = self.function_create_button("Complete Cycle", column_1_layout_internal, self.function_Multiple_Train_Buttons)
        
    #     # Create Label for available datasets
    #     self.function_create_label("NUMERAI Available Datasets", column_1_layout_internal)
       
    #     #Create a listwidget to house the Datasets available to download
    #     self.list_widget_all_datasets = QListWidget()
    #     self.list_widget_all_datasets.setSelectionMode(QAbstractItemView.MultiSelection)
    #     column_1_layout_internal.addWidget(self.list_widget_all_datasets)
        
    #     # Create download button
    #     self.button_download_data_set_selected = self.function_create_button("Downloaded Selected Datasets", column_1_layout_internal, self.function_button_download_selected_dataset)

    #     # Create label for downloaded datasets    
    #     self.function_create_label("Datasets Downloadeds", column_1_layout_internal)
        
    #     #Create a listwidget to house the Datasets available to download
    #     self.list_Widget_Availabile_Downloaded_Datasets = QListWidget()
    #     self.function_update_downloaded_datasets_list() #Fuction
    #     column_1_layout_internal.addWidget(self.list_Widget_Availabile_Downloaded_Datasets)

    #     # Section for Features Dataset List       
    #     self.function_create_label("Features List - Select One", column_1_layout_internal)
    #     self.list_Widget_Features_Downloaded_Datasets= QListWidget()
    #     column_1_layout_internal.addWidget(self.list_Widget_Features_Downloaded_Datasets)
        
    #     self.list_Widget_Features_Downloaded_Datasets.itemClicked.connect(self.function_display_feature_list)  

    #     # Section for Content of Feature List      
    #     self.function_create_label("Features List Contents - Select One", column_1_layout_internal)
    #     self.list_widget_features_content= QListWidget()
    #     column_1_layout_internal.addWidget(self.list_widget_features_content)

    #     # Section for Metadata Display
    #     self.metadata_widget = QWidget()  
    #     metadata_layout = QVBoxLayout()    
    #     self.metadata_widget.setLayout(metadata_layout)
    #     column_1_layout_internal.addWidget(self.metadata_widget)  
        
    #     self.list_widget_features_content.itemClicked.connect(self.function_handle_feature_list_change)
        
    #     # Add internal layout to the main layout
    #     self.Column_1_Train_set_layout.addLayout(column_1_layout_internal)
    
    #     self.function_hide_widgets_in_layout(self.Column_1_Train_set_layout)
        
            
    # def initialize_Column_2_View_Data_Set_Layout(self) -> None:
    #     """
    #     Initializes the layout for Column 2 to display training and validation datasets,
    #     including a metadata section for selected datasets.
    #     """
    #     column_2_layout_internal = QVBoxLayout()
        
    #     # Section for Training Dataset List       
    #     self.function_create_label("Training Dataset List:", column_2_layout_internal)
    #     self.list_widget_train_downloaded_datasets = QListWidget()
    #     column_2_layout_internal.addWidget(self.list_widget_train_downloaded_datasets)
        
    #     # List of available training methods Q ComboBox
    #     self.function_create_label("Select Training Module:", column_2_layout_internal)
    #     self.training_methods = ["LGBMRegressor", "HistGradientBoostingRegressor"]  
    #     self.training_method_combo = QComboBox()
    #     self.training_method_combo.addItems(self.training_methods)
    #     column_2_layout_internal.addWidget(self.training_method_combo)

    #     # Connect signal for combobox selection change
    #     self.training_method_combo.currentIndexChanged.connect(self.function_training_method_changed)
        
    #     # Create layout for LGBMRegressor hyperparameters
    #     self.hyperparameters_layout_LGM = QVBoxLayout()
    #     widgets_LGM = []  # List to store widgets for LGBMRegressor parameters
        
    #     self.function_create_label("Number of boosted trees to fit", self.hyperparameters_layout_LGM)
    #     self.n_estimators = QLineEdit()
    #     self.n_estimators.setPlaceholderText("2000")
    #     self.hyperparameters_layout_LGM.addWidget(self.n_estimators)
    #     widgets_LGM.append(self.n_estimators)
        
    #     self.function_create_label("Boosting learning rate", self.hyperparameters_layout_LGM)
    #     self.learning_rate = QLineEdit()
    #     self.learning_rate.setPlaceholderText("0.01")
    #     self.hyperparameters_layout_LGM.addWidget(self.learning_rate)
    #     widgets_LGM.append(self.learning_rate)
        
    #     self.function_create_label("Maximum tree depth for base learners", self.hyperparameters_layout_LGM)
    #     self.max_depth = QLineEdit()
    #     self.max_depth.setPlaceholderText("5")
    #     self.hyperparameters_layout_LGM.addWidget(self.max_depth)
    #     widgets_LGM.append(self.max_depth)
        
    #     self.function_create_label("Maximum tree leaves for base learners", self.hyperparameters_layout_LGM)
    #     self.num_leaves = QLineEdit()
    #     self.num_leaves.setPlaceholderText("2**5-1")
    #     self.hyperparameters_layout_LGM.addWidget(self.num_leaves)
    #     widgets_LGM.append(self.num_leaves)
        
    #     self.function_create_label("Subsample ratio of columns when constructing each tree", self.hyperparameters_layout_LGM)
    #     self.colsample_bytree = QLineEdit()
    #     self.colsample_bytree.setPlaceholderText("0.1")
    #     self.hyperparameters_layout_LGM.addWidget(self.colsample_bytree)
    #     widgets_LGM.append(self.colsample_bytree)
        
        
    #     column_2_layout_internal.addLayout(self.hyperparameters_layout_LGM)

    #     # Create layout for Scitkit hyperparameters
    #     self.hyperparameters_layout_HistGradientBoostingRegressor = QVBoxLayout()
    #     widgets_HGBR =[]   # Dictionary to store widgets for Scikit regressor parameters

    #     self.function_create_label("Learning Rate", self.hyperparameters_layout_HistGradientBoostingRegressor)
    #     self.learning_rate_sci = QLineEdit()
    #     self.learning_rate_sci.setPlaceholderText("0.1")
    #     self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.learning_rate_sci)
    #     widgets_HGBR.append(self.learning_rate_sci)
        
    #     self.function_create_label("Max Number of Trees", self.hyperparameters_layout_HistGradientBoostingRegressor)
    #     self.max_iter_sci = QLineEdit()
    #     self.max_iter_sci.setPlaceholderText("100")
    #     self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_iter_sci)
    #     widgets_HGBR.append(self.max_iter_sci )        
        
    #     self.function_create_label("Max Num of leaves per Tree", self.hyperparameters_layout_HistGradientBoostingRegressor)
    #     self.max_leaf_nodes_sci = QLineEdit()
    #     self.max_leaf_nodes_sci.setPlaceholderText("31")
    #     self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_leaf_nodes_sci)
    #     widgets_HGBR.append(self.max_leaf_nodes_sci)    
        
    #     self.function_create_label("Max Depth of Tree", self.hyperparameters_layout_HistGradientBoostingRegressor)
    #     self.max_depth_sci = QLineEdit()
    #     self.max_depth_sci.setPlaceholderText("None")
    #     self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_depth_sci)
    #     widgets_HGBR.append(self.max_depth_sci)    
        
    #     self.function_create_label("Proportion of random chosen features in each node split", self.hyperparameters_layout_HistGradientBoostingRegressor)
    #     self.max_features_sci = QLineEdit()
    #     self.max_features_sci.setPlaceholderText("0.1")
    #     self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_features_sci)
    #     widgets_HGBR.append(self.max_features_sci)    
        
    #     # Initially hide the layout for "Other" hyperparameters
    #     for i in range(self.hyperparameters_layout_HistGradientBoostingRegressor.count()):
    #         widget = self.hyperparameters_layout_HistGradientBoostingRegressor.itemAt(i).widget()
    #         if widget:
    #             widget.hide()
        
    #     column_2_layout_internal.addLayout(self.hyperparameters_layout_HistGradientBoostingRegressor)

    #     # Create download button
    #     self.button_download_data_set_selected = self.function_create_button("Train", column_2_layout_internal, self.function_Button_Train_Model)
        
    #     self.function_create_label("First 50 Rows Training", column_2_layout_internal)
    #     self.table_widget_train_dataset = QTableWidget()
    #     column_2_layout_internal.addWidget(self.table_widget_train_dataset)
        
    #     # Set the layout for the column
    #     self.Column_2_view_data_set_layout.addLayout(column_2_layout_internal)
    #     self.function_hide_widgets_in_layout(self.Column_2_view_data_set_layout)
        
    # def initialize_Column_3_Modelling_Data_Set_Layout(self)-> None:
    #     """
    #     Initializes the layout for Column 3 to display validation datasets and associated metadata,
    #     as well as a table for dataset validation.
    #     """
    #     # Create the internal layout for Column 3
    #     column_3_layout_internal = QVBoxLayout()
        
    #     # Section for Validation Dataset List      
    #     self.function_create_label("Validation list:", column_3_layout_internal)
    #     self.list_widget_validation_downloaded_datasets= QListWidget()        
    #     column_3_layout_internal.addWidget(self.list_widget_validation_downloaded_datasets)
        
    #     # Create Validate button
    #     self.button_validate_data_Set_selected = self.function_create_button("Validate", column_3_layout_internal, self.function_Button_Validate_Model)
      
    #     self.function_create_label("First 50 Rows Validation Table", column_3_layout_internal)
    #     self.table_widget_validation_dataset = QTableWidget()
    #     column_3_layout_internal.addWidget(self.table_widget_validation_dataset)
        
    #     self.function_create_label("Validation Results Table", column_3_layout_internal)
    #     self.table_widget_validation_results = QTableWidget()
    #     column_3_layout_internal.addWidget(self.table_widget_validation_results)
                
    #     # Set the layout for the column
    #     self.Column_3_modelling_data_set_layout.addLayout(column_3_layout_internal)
    #     self.function_hide_widgets_in_layout(self.Column_3_modelling_data_set_layout)
        
    # def initialize_column_4_live_model_layout(self)-> None:
        # column_4_layout_internal = QVBoxLayout()
        
        # # Section for Validation list
        # self.function_create_label("Performance metrics List:", column_4_layout_internal)
        # self.list_Widget_meta_model_datasets = QListWidget()
        # column_4_layout_internal.addWidget(self.list_Widget_meta_model_datasets)
        
        # # Create Peformance Evaluation button
        # self.button_performance_evaluation_data_set_selected = self.function_create_button("Performance Evaluation", column_4_layout_internal, self.function_Button_Evaluate_Peforance_Model)
        
        # #Table for the meta model perforance metricx
        # self.function_create_label("Meta Model Performance Table", column_4_layout_internal)
        # self.table_widget_metamodel_performance_file = QTableWidget()
        # column_4_layout_internal.addWidget(self.table_widget_metamodel_performance_file)
        
        # self.function_create_label("Validation Results Table", column_4_layout_internal)
        # self.table_widget_performance_results = QTableWidget()
        # column_4_layout_internal.addWidget(self.table_widget_performance_results)
          
        # # Section for Live Prediction List
        # self.function_create_label("Live Prediction List:", column_4_layout_internal)
        # self.list_Widget_Live_Prediction_Datasets = QListWidget()
        # column_4_layout_internal.addWidget(self.list_Widget_Live_Prediction_Datasets)        
        
        # # Create Peformance Evaluation button
        # self.button_Live_Prediction_Data_Set_Selected = self.function_create_button("Live Prediction:", column_4_layout_internal, self.function_Button_Live_predictions_Model)     
   
        # # Set the layout for the column
        # self.Column_4_live_model_layout.addLayout(column_4_layout_internal)
        # self.function_hide_widgets_in_layout(self.Column_4_live_model_layout)
        
    # def initialize_column_5_debugger_layout(self)-> None:
    #     # Add the QTextEdit widget to the debugger layout in Column 4
    #     self.Column_5_debugger_layout.addWidget(self.terminal_widget)
    
    
        # def function_cleanup_sandbox(self)->None:
    #     try:
    #         # Delete sandbox directory and its contents
    #         shutil.rmtree(self.sandbox_dir)
    #         logging.info("Sandbox directory cleanup completed.")
    #     except FileNotFoundError:
    #         logging.warning("Sandbox directory not found.")
    #     except PermissionError:
    #         logging.error("Permission denied. Unable to delete sandbox directory.")
    #     except Exception as e:
    #         logging.error(f"An error occurred during cleanup: {e}")
    #     else:
    #         print("Cleanup completed.")
 