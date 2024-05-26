from numerapi import NumerAPI
import json
import pandas as pd
import os
import re
from lightgbm import LGBMRegressor
import cloudpickle
import sys
import certifi
import urllib3
import logging
import platform
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)

# import the 2 scoring functions
from numerai_tools.scoring import numerai_corr, correlation_contribution

from PySide6.QtGui import QPixmap, QValidator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QListWidget, QTableWidgetItem, QListWidgetItem, QTableWidget, QSizePolicy, QAbstractItemView, QSizePolicy, QComboBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QFileDialog,
    QLabel, QPushButton, QMessageBox,QStackedWidget
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
        
        #Define Feature Set and File Parameters
        self.selected_feature_file = None
        self.selected_feature_set= None
        # self.selected_feature_sets_all_features = None
        self.num_of_features = None
        self.live_features_stored = None
         
        self.initialize_all_layouts()
        self.initialize_Column_0_menu_Layout()
        self.initialize_Column_2_Body_model_layout()
   
        #Connect List Widget to NumerAi function
        self.function_Download_all_available_datasets_numerAi()
        
        # Create instances of  redirectors
        self.stdout_redirector = StdoutRedirector(text_widget=self.terminal_widget)
        self.stderr_redirector = StderrRedirector(text_widget=self.terminal_widget)
        
        # Redirect sys.stdout and sys.stderr to your redirectors
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        # Initialize a dictionary to store trained models
        self.trained_models = {}
        
        
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
        self.button_train_model_step_1 = self.function_create_button("Datasets", column_0_layout_internal, self.hide_other_layouts, button_style)
        self.button_validate_model_step_2= self.function_create_button("Train", column_0_layout_internal, self.hide_other_layouts,button_style)
        self.button_live_model_step_4 = self.function_create_button("Muti-Automate", column_0_layout_internal, self.hide_other_layouts,button_style)
        self.button_peformance_metrics_step_3 = self.function_create_button("Tables", column_0_layout_internal, self.hide_other_layouts,button_style)

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
            self.button_label = QLabel("Datasets")
            self.button_label.setStyleSheet("font-size: 19px; color: #000000; font-family: Arial;")
            column_2_layout_internal_Header.addWidget(self.button_label)

            # Create the body widget
            self.body_widget = QStackedWidget()
            self.body_widget.setStyleSheet("background-color: #DEDEDC;")
            self.Column_2_Body_model_layout.addWidget(self.body_widget)

            # Create and add different pages to the stacked widget
            self.column_2_layout_internal_dataset = self.set_layout_dataset()
            self.column_2_layout_internal_train = self.set_layout_to_train()
            self.column_2_layout_internal_Live = self.set_layout_to_Automate()
            self.column_2_layout_internal_Peforomance = self.set_layout_to_peformance()

            self.body_widget.addWidget(self.column_2_layout_internal_dataset)
            self.body_widget.addWidget(self.column_2_layout_internal_train)
            self.body_widget.addWidget(self.column_2_layout_internal_Live)
            self.body_widget.addWidget(self.column_2_layout_internal_Peforomance)
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
    
            
    def set_layout_dataset(self):
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
            background-color: #DEDEDC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        
        list_style = """
            background-color: #FCFCFC;
            color: #000000;
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

        layout.addLayout(left_column_layout)
        
        # Right column
        right_column_layout = QVBoxLayout()   

        self.function_create_label("Features List - Select One", right_column_layout,label_style)
        
        self.list_Widget_Features_Downloaded_Datasets= QListWidget()
        self.list_Widget_Features_Downloaded_Datasets.setStyleSheet(list_style)
        right_column_layout.addWidget(self.list_Widget_Features_Downloaded_Datasets)
        self.list_Widget_Features_Downloaded_Datasets.itemClicked.connect(self.function_display_feature_list)  

        self.function_create_label("Features List Contents - Select One", right_column_layout,label_style)
        
        self.list_widget_features_content= QListWidget()
        self.list_widget_features_content.setStyleSheet(list_style)
        right_column_layout.addWidget(self.list_widget_features_content)
        self.list_widget_features_content.itemClicked.connect(self.function_handle_feature_list_change)

        # Section for Training Dataset List       
        self.function_create_label("Training Dataset List:", right_column_layout,label_style)
        self.list_widget_train_downloaded_datasets = QListWidget()
        self.list_widget_train_downloaded_datasets.setStyleSheet(list_style)
        right_column_layout.addWidget(self.list_widget_train_downloaded_datasets)
        
        # Section for Validation Dataset List      
        self.function_create_label("Validation list:", right_column_layout,label_style)
        self.list_widget_validation_downloaded_datasets= QListWidget()     
        self.list_widget_validation_downloaded_datasets.setStyleSheet(list_style)   
        right_column_layout.addWidget(self.list_widget_validation_downloaded_datasets)
        
        # Section for Validation list
        self.function_create_label("Performance metrics List:", right_column_layout,label_style)
        self.list_Widget_meta_model_datasets = QListWidget()
        self.list_Widget_meta_model_datasets.setStyleSheet(list_style)
        right_column_layout.addWidget(self.list_Widget_meta_model_datasets)
                  
        self.metadata_widget = QWidget()  
        metadata_layout = QVBoxLayout()    
        self.metadata_widget.setLayout(metadata_layout)
        right_column_layout.addWidget(self.metadata_widget)
               
        layout.addLayout(right_column_layout)        
           
        return page_widget
    
    

    def set_layout_to_train(self):
        button_style = """
            background-color: #416096;
            color: #F9FFFF;
            font-family: Arial;
            font-size: 12;
        """
        label_style = """
            background-color: #DEDEDC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
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

        page_widget = QWidget()
        layout = QHBoxLayout(page_widget)
        
        # Left column
        left_column_layout = QVBoxLayout()

        # List of available training methods Q ComboBox
        self.function_create_label("Select Training Module:", left_column_layout,label_style)
        self.training_methods = ["LGBMRegressor", "HistGradientBoostingRegressor"]  
        self.training_method_combo = QComboBox()
        self.training_method_combo.setStyleSheet(combobox_style)
        self.training_method_combo.addItems(self.training_methods)
        left_column_layout.addWidget(self.training_method_combo)

        # Connect signal for combobox selection change
        self.training_method_combo.currentIndexChanged.connect(self.function_training_method_changed)
        
        # Create layout for LGBMRegressor hyperparameters
        self.hyperparameters_layout_LGM = QVBoxLayout()
        widgets_LGM = []  # List to store widgets for LGBMRegressor parameters
        
    
        self.function_create_label("Number of boosted trees to fit", self.hyperparameters_layout_LGM,label_style)
        
        # Create a QLineEdit with the custom validator
        self.n_estimators = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.n_estimators.setValidator(validator)


        # self.n_estimators = QLineEdit()
        self.n_estimators.setStyleSheet(lineedit_style)
        self.n_estimators.setPlaceholderText("Please input Integer values like 2000 For multiple values, separate them with commas like 2000, 3000")
        self.hyperparameters_layout_LGM.addWidget(self.n_estimators)
        widgets_LGM.append(self.n_estimators)
        
        
        self.function_create_label("Boosting learning rate", self.hyperparameters_layout_LGM,label_style)
        
        # Create a QLineEdit with the custom validator
        self.learning_rate = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.learning_rate.setValidator(validator)
        
        self.learning_rate.setStyleSheet(lineedit_style)
        self.learning_rate.setPlaceholderText("Please input float values like 0.01 For multiple values, separate them with commas like 0.01, 0.02")
        self.hyperparameters_layout_LGM.addWidget(self.learning_rate)
        widgets_LGM.append(self.learning_rate)
        
        self.function_create_label("Maximum tree depth for base learners", self.hyperparameters_layout_LGM,label_style)
        
        self.max_depth = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.max_depth.setValidator(validator)
        
        self.max_depth.setStyleSheet(lineedit_style)
        self.max_depth.setPlaceholderText("Please input Integer values like 5 For multiple values, separate them with commas like 5,6")
        self.hyperparameters_layout_LGM.addWidget(self.max_depth)
        widgets_LGM.append(self.max_depth)
        
        self.function_create_label("Maximum tree leaves for base learners", self.hyperparameters_layout_LGM,label_style)
        
        self.num_leaves = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.num_leaves.setValidator(validator)
        
        self.num_leaves.setStyleSheet(lineedit_style)
        self.num_leaves.setPlaceholderText("Please input Integer values like 30 For multiple values, separate them with commas like 30,31")
        self.hyperparameters_layout_LGM.addWidget(self.num_leaves)
        widgets_LGM.append(self.num_leaves)
        
        self.function_create_label("Subsample ratio of columns when constructing each tree", self.hyperparameters_layout_LGM,label_style)
        
        self.colsample_bytree = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.colsample_bytree.setValidator(validator)
        
        self.colsample_bytree.setStyleSheet(lineedit_style)
        self.colsample_bytree.setPlaceholderText("Please input float values like 0.1 For multiple values, separate them with commas like 0.1, 0.25")
        self.hyperparameters_layout_LGM.addWidget(self.colsample_bytree)
        widgets_LGM.append(self.colsample_bytree)
        
        left_column_layout.addLayout(self.hyperparameters_layout_LGM)

        # Create layout for Scitkit hyperparameters
        self.hyperparameters_layout_HistGradientBoostingRegressor = QVBoxLayout()
        
        widgets_HGBR =[]   # Dictionary to store widgets for Scikit regressor parameters

        self.function_create_label("Learning Rate", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        
        self.learning_rate_sci = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.learning_rate_sci.setValidator(validator)
        
        self.learning_rate_sci.setStyleSheet(lineedit_style)
        self.learning_rate_sci.setPlaceholderText("Please input float values like 0.01 For multiple values, separate them with commas like 0.01, 0.02")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.learning_rate_sci)
        widgets_HGBR.append(self.learning_rate_sci)
        
        self.function_create_label("Max Number of Trees", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        
        self.max_iter_sci = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.max_iter_sci.setValidator(validator)
        
        self.max_iter_sci.setStyleSheet(lineedit_style)    
        self.max_iter_sci.setPlaceholderText("Please input Integer values like 2000 For multiple values, separate them with commas like 2000, 30")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_iter_sci)
        widgets_HGBR.append(self.max_iter_sci )        
        
        self.function_create_label("Max Num of leaves per Tree", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        
        self.max_leaf_nodes_sci = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.max_leaf_nodes_sci.setValidator(validator)
        
        self.max_leaf_nodes_sci.setStyleSheet(lineedit_style)  
        self.max_leaf_nodes_sci.setPlaceholderText("Please input Integer values like 31 For multiple values, separate them with commas like 30,31")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_leaf_nodes_sci)
        widgets_HGBR.append(self.max_leaf_nodes_sci)    
        
        self.function_create_label("Max Depth of Tree", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        
        self.max_depth_sci = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.max_depth_sci.setValidator(validator)
        
        self.max_depth_sci.setStyleSheet(lineedit_style)  
        self.max_depth_sci.setPlaceholderText("Please input Integer values like 5 For multiple values, separate them with commas like 5,6")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_depth_sci)
        widgets_HGBR.append(self.max_depth_sci)    
        
        self.function_create_label("Proportion of random chosen features in each node split", self.hyperparameters_layout_HistGradientBoostingRegressor,label_style)
        
        self.max_features_sci = QLineEdit()
        validator = DelimitedValidator()  # Adjust delimiter and additional_delimiter if needed
        self.max_features_sci.setValidator(validator)
        
        self.max_features_sci.setStyleSheet(lineedit_style)  
        self.max_features_sci.setPlaceholderText("Please input float values like 0.1 For multiple values, separate them with commas like 0.1, 0.2")
        self.hyperparameters_layout_HistGradientBoostingRegressor.addWidget(self.max_features_sci)
        widgets_HGBR.append(self.max_features_sci)    
        
        # Initially hide the layout for "Other" hyperparameters
        for i in range(self.hyperparameters_layout_HistGradientBoostingRegressor.count()):
            widget = self.hyperparameters_layout_HistGradientBoostingRegressor.itemAt(i).widget()
            if widget:
                widget.hide()
        
        left_column_layout.addLayout(self.hyperparameters_layout_HistGradientBoostingRegressor)

        # Create download button
        self.button_download_data_set_selected = self.function_create_button("Train Multi Models", left_column_layout, self.function_Multiple_Train_Buttons, button_style) 
        
        left_column_layout.setAlignment(Qt.AlignTop) 
        layout.addLayout(left_column_layout)  
        
        # Right column
        right_column_layout = QVBoxLayout()

        # Create QLabel for displaying the image
        image_label = QLabel()
        pixmap = QPixmap("C:/Users/aat2g/OneDrive/Documents/13. Personal Project/7. NumerAi Test/Numerai-modelling-app/learning.jpg")

        # Scale pixmap to fit the size of the QLabel
        image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        image_label.setAlignment(Qt.AlignCenter)

        # Set maximum size policy for the image label (occupies maximum 40% of space)
        image_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        image_label.setMaximumSize(800, 800)  # Example size, adjust as needed

        # Add the image QLabel to the right layout
        right_column_layout.addWidget(image_label, alignment=Qt.AlignCenter)

        layout.addLayout(right_column_layout)
        
        return page_widget


    def set_layout_to_peformance(self):
            
        label_style = """
            background-color: #DEDEDC;
            color: #000000;
            font-family: Noto Sans Tangsa;
            font-size: 12px;
        """
        Table_style = """
        color: #000000;
        font-size: 10px;
        font-family: Arial;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 2px;
        padding: 2px;
        selection-background-color: #BFE6FF;
        selection-color: #000000;
        """    
           
        page_widget = QWidget()
        layout = QHBoxLayout(page_widget)

        # Left column
        left_column_layout = QVBoxLayout()
        
        self.function_create_label("First 50 Rows Training", left_column_layout,label_style)
        self.table_widget_train_dataset = QTableWidget()
        self.table_widget_train_dataset .setStyleSheet(Table_style)
        left_column_layout.addWidget(self.table_widget_train_dataset)
        
        self.function_create_label("First 50 Rows Validation Table", left_column_layout,label_style)
        self.table_widget_validation_dataset = QTableWidget()
        self.table_widget_validation_dataset.setStyleSheet(Table_style)
        left_column_layout.addWidget(self.table_widget_validation_dataset)
        
        layout.addLayout(left_column_layout)
        
        # Right column
        right_column_layout = QVBoxLayout()   
        
        self.function_create_label("Validation Results Table", right_column_layout,label_style)
        self.table_widget_validation_results = QTableWidget()
        self.table_widget_validation_results.setStyleSheet(Table_style)
        right_column_layout.addWidget(self.table_widget_validation_results)
        
        #Table for the meta model perforance metricx
        self.function_create_label("Meta Model Performance Table", right_column_layout,label_style)
        self.table_widget_metamodel_performance_file = QTableWidget()
        self.table_widget_metamodel_performance_file.setStyleSheet(Table_style)
        right_column_layout.addWidget(self.table_widget_metamodel_performance_file)
        
        layout.addLayout(right_column_layout)       
                
        return page_widget
    
    
    
    def set_layout_to_Automate(self):
        Table_style = """
        color: #000000;
        font-size: 10px;
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
        self.table_widget_multi_results = QTableWidget()
        self.table_widget_multi_results.setStyleSheet(Table_style)
        layout.addWidget(self.table_widget_multi_results)
        

        # Create a horizontal layout for the graphs
        graph_layout = QHBoxLayout()
        
        # Create the first graph widget and its toolbar
        self.graph_widget_1 = FigureCanvas(Figure(figsize=(7, 5)))  # Set initial size of the figure
        self.graph_widget_1_toolbar = NavigationToolbar(self.graph_widget_1, self)
        self.graph_widget_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        graph_layout.addWidget(self.graph_widget_1)
        graph_layout.addWidget(self.graph_widget_1_toolbar)

        # Create the second graph widget and its toolbar
        self.graph_widget_2 = FigureCanvas(Figure(figsize=(7, 5)))  # Set initial size of the figure
        self.graph_widget_2_toolbar = NavigationToolbar(self.graph_widget_2, self)
        self.graph_widget_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        graph_layout.addWidget(self.graph_widget_2)
        graph_layout.addWidget(self.graph_widget_2_toolbar)
        
        # Add the graph layout to the main layout
        layout.addLayout(graph_layout)

        # Connect table selection signal to the update_graphs function
        self.table_widget_multi_results.itemSelectionChanged.connect(self.function_update_graphs)
        
        return page_widget
  
    def hide_other_layouts(self):
        """
        Hide all the layouts related to different steps except the one associated with the clicked button.
        """
        sender_button = self.sender()
        sender_name = sender_button.text()
        
        # Determine which layout widgets to show based on the clicked button's name
        if sender_name == "Datasets":
            if self.body_widget.currentIndex() != 0:  # Check if current index is not already 0
                self.button_label.setText(sender_name)
                self.body_widget.setCurrentIndex(0)
                    
        elif sender_name == "Train":
            # Check if current index is not already 1
            if self.body_widget.currentIndex() != 1:
                # Check if any of the required lists are empty or no items are selected
                if (not self.list_Widget_Features_Downloaded_Datasets.selectedItems() or
                    not self.list_widget_validation_downloaded_datasets.selectedItems() or
                    not self.list_widget_train_downloaded_datasets.selectedItems() or
                    not self.list_Widget_meta_model_datasets.selectedItems() or
                    not self.list_widget_features_content.selectedItems()):
                    
                    # Inform the user to download and select items from the missing lists
                    missing_lists = []
                    if not self.list_Widget_Features_Downloaded_Datasets.selectedItems():
                        missing_lists.append("Features List")
                    if not self.list_widget_validation_downloaded_datasets.selectedItems():
                        missing_lists.append("Validation Dataset List")
                    if not self.list_widget_train_downloaded_datasets.selectedItems():
                        missing_lists.append("Training Dataset List")
                    if not self.list_Widget_meta_model_datasets.selectedItems():
                        missing_lists.append("Performance Metrics List")
                    if not self.list_widget_features_content.selectedItems():
                        missing_lists.append("Features List Contents")
                        
                    QMessageBox.warning(self, "Missing Selection",
                                        f"Please download and select items from the following lists: {', '.join(missing_lists)}")
                else:
                    self.button_label.setText(sender_name)
                    self.body_widget.setCurrentIndex(1)
            
        elif sender_name == "Tables":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(3)

        elif sender_name == "Muti-Automate":
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(2)


    def function_update_graphs(self):
        # Check if self.trained_models is not empty
        if self.trained_models:
            # Get the selected row indexes
            selected_indexes = self.table_widget_multi_results.selectedIndexes()
            if selected_indexes:
                selected_row = selected_indexes[0].row()  # Assuming single selection
                (results_df, model, validation, Performance_validation, per_era_corr, per_era_mmc) = self.trained_models[selected_row]

                # Clear existing plots from the Figure objects associated with graph_widget_1 and graph_widget_2
                self.graph_widget_1.figure.clear()
                self.graph_widget_2.figure.clear()

                # Plot cumulative validation CORR
                ax1 = self.graph_widget_1.figure.add_subplot(111)
                per_era_corr.cumsum().plot(
                    ax=ax1,
                    title="Cumulative Validation CORR",
                    kind="line",
                    legend=False
                )
                # Use subplots_adjust to manually set margins
                self.graph_widget_1.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

                # Plot cumulative validation MMC
                ax2 = self.graph_widget_2.figure.add_subplot(111)
                per_era_mmc.cumsum().plot(
                    ax=ax2,
                    title="Cumulative Validation MMC",
                    kind="line",
                    legend=False
                )
                # Use subplots_adjust to manually set margins
                self.graph_widget_2.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

                # Redraw the graph widgets to reflect the changes
                self.graph_widget_1.draw()
                self.graph_widget_2.draw()
                
                # Populate table widget
                self.setup_validation_results_table(validation)
                        
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
                print(f"Error loading dataset: {e}")
        else:
            self.selected_feature_set = None
            self.selected_feature_file = None
            self.num_of_features = 0

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
            self.function_create_label(f"Feature List File: {self.selected_feature_file}", layout, label_style)
        else:
            self.function_create_label("Feature List File: N/A", layout, label_style)

        if self.selected_feature_set:
            self.function_create_label(f"Feature List Selected: {self.selected_feature_set}", layout, label_style)
        else:
            self.function_create_label("Feature List Selected: N/A", layout, label_style)

        if self.num_of_features:
            self.function_create_label(f"Number of Features: {self.num_of_features}", layout, label_style)
        else:
            self.function_create_label("Number of Features: N/A", layout, label_style)
            
                
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
                
                
    def function_update_downloaded_datasets_list(self,folder_path) -> None:
        """
        Updates the list of downloaded datasets.
        """
        # Clear the lists if they exist
        if hasattr(self, 'list_Widget_Availabile_Downloaded_Datasets'):
            self.list_Widget_Availabile_Downloaded_Datasets.clear()
        if hasattr(self, 'list_Widget_Features_Downloaded_Datasets'):
            self.list_Widget_Features_Downloaded_Datasets.clear()
            
        if hasattr(self, 'list_widget_validation_downloaded_datasets'):
            self.list_widget_validation_downloaded_datasets.clear()
        if hasattr(self, 'list_widget_train_downloaded_datasets'):
            self.list_widget_train_downloaded_datasets.clear()
        if hasattr(self, 'list_Widget_meta_model_datasets'):
            self.list_Widget_meta_model_datasets.clear()
        if hasattr(self, 'list_widget_features_content'):
            self.list_widget_features_content.clear()        
            self.function_handle_feature_list_change(None)

        try:
            for root, dirs, files in os.walk(folder_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    for file in os.listdir(dir_path):
                        if file.endswith(('.csv', '.parquet', '.json', '.xlsx', '.db', '.sqlite', '.sqlite3')):
                            file_path = os.path.join(dir_path, file)
                            file_path = file_path.replace('\\', '/')
                            relative_path = os.path.relpath(file_path, folder_path).replace('\\', '/')
                            self.list_Widget_Availabile_Downloaded_Datasets.addItem(relative_path)
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
             
    def function_Multiple_Train_Buttons(self) -> None: 
        # Retrieve selected feature sets
        selected_feature_sets = self.get_selected_feature_sets()
        if selected_feature_sets is None:
            return
        
        # Load training data
        train = self.load_training_data(selected_feature_sets)
        if train is None:
            return
        
        # Load performance metric file
        validation_file_path = self.verify_load_validation_file_selected()
        if validation_file_path is None:
            return
        
        # Load performance metric file
        Performance_validation_file_path = self.verify_load_performance_metric_file_selected()
        if Performance_validation_file_path is None:
            return

        # Verify modelling method algorithm selected by user 
        selected_method = self.training_method_combo.currentText()
        if selected_method == "LGBMRegressor":
            hyperparameters_grid = self.create_lgbm_hyperparameter_grid()
        elif selected_method == "HistGradientBoostingRegressor":
            hyperparameters_grid = self.create_HistGradientBoostingRegressor_hyperparameter_grid()
        else:
            QMessageBox.warning(self, "Invalid Input", "Please select a valid training method.")
            return
        
        trained_models_table_this_round_only = {} #this copies trained  model for the table 
        
        try:
            if hyperparameters_grid:  # Check if hyperparameters_grid is not None or empty
                for i in range(len(next(iter(hyperparameters_grid.values())))):
                    try:  
                        hyperparameters_dict = {key: hyperparameters_grid[key][i] for key in hyperparameters_grid}

                        model = self.train_lgb_model_with_hyperparameters(hyperparameters_dict, selected_feature_sets, train)
                    except Exception as e:
                        print(f"Error training model with hyperparameters {hyperparameters_dict}: {e}")
                        continue    
                        
                    try:                    
                        # Load validation data
                        validation = self.load_validation_data(selected_feature_sets,validation_file_path)
                        if validation is None:
                            return

                        validation = validation[validation["data_type"] == "validation"]
                        del validation["data_type"]

                        # Eras are 1 week apart, but targets look 20 days (or 4 weeks/eras) into the future,
                        # so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
                        last_train_era = int(train["era"].unique()[-1])
                        eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
                        validation = validation[~validation["era"].isin(eras_to_embargo)]
                        
                        self.fuction_parquet_data_into_table(validation,self.table_widget_validation_dataset)

                        # Generate predictions against the out-of-sample validation features
                        # This will take a few minutes
                        validation["prediction"] = model.predict(validation[selected_feature_sets])
                        print(validation[["era", "prediction", "target"]])
                    
                    except Exception as e:
                        print(f"Error during validation or prediction: {e}")
                        continue
            
                    try:
                        # Load performance metric file
                        Performance_validation = self.load_performance_metric_file(validation,Performance_validation_file_path)
                        if Performance_validation is None:
                            return
                        
                        # Compute the per-era corr between our predictions and the target values
                        per_era_corr = Performance_validation.groupby("era").apply(
                            lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
                        )

                        per_era_mmc = Performance_validation.dropna().groupby("era").apply(
                            lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
                        )
                        
                    except Exception as e:
                        print(f"Error computing performance metrics: {e}")
                        continue
                    
                    try:
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
                            'n_estimators': [hyperparameters_dict.get('n_estimators')],
                            'learning_rate': [hyperparameters_dict.get('learning_rate')],
                            'max_depth': [hyperparameters_dict.get('max_depth')],
                            'num_leaves': [hyperparameters_dict.get('num_leaves')],
                            'colsample_bytree': [hyperparameters_dict.get('colsample_bytree')],
                            'max_iter': [hyperparameters_dict.get('max_iter')],
                            'max_leaf_nodes': [hyperparameters_dict.get('max_leaf_nodes')],
                            'max_features': [hyperparameters_dict.get('max_features')],
                            "corr_mean": [corr_mean],
                            "mmc_mean": [mmc_mean],
                            "corr_std": [corr_std],
                            "mmc_std": [mmc_std],
                            "corr_sharpe": [corr_sharpe],
                            "mmc_sharpe": [mmc_sharpe],
                            "corr_max_drawdown": [corr_max_drawdown],
                            "mmc_max_drawdown": [mmc_max_drawdown]
                        })

                        if model is not None:
                            # Determine the next number for self.trained_models
                            # If self.trained_models is empty, max(..., default=-1) returns -1, so next_number becomes 0
                            next_number = max(self.trained_models.keys(), default=-1) + 1
                            
                            # Determine the next number for trained_models_table_this_round_only
                            # If trained_models_table_this_round_only is empty, max(..., default=-1) returns -1, so next_number_this_round becomes 0
                            next_number_this_round = max(trained_models_table_this_round_only.keys(), default=-1) + 1
                            
                            # Store the hyperparameters dictionary and model
                            model_data = (results_df, model, validation, Performance_validation, per_era_corr, per_era_mmc)
                            self.trained_models[next_number] = model_data
                            trained_models_table_this_round_only[next_number_this_round] = model_data

                    except Exception as e:
                        print(f"Error storing model data: {e}")
                        continue


                # Get the current row count to start adding new rows
                current_row_count = self.table_widget_multi_results.rowCount()

                # Set the column count and headers if not already set
                if self.table_widget_multi_results.columnCount() == 0:
                    headers = list(results_df.columns) + ['']
                    self.table_widget_multi_results.setColumnCount(len(headers))
                    self.table_widget_multi_results.setHorizontalHeaderLabels(headers)
                else:
                    headers = [self.table_widget_multi_results.horizontalHeaderItem(i).text() for i in range(self.table_widget_multi_results.columnCount())]

                # Iterate over trained models dictionary with an index starting from the current row count
                for row_num, (index, (results_df, model, validation, Performance_validation,_,_)) in enumerate(trained_models_table_this_round_only.items(), start=current_row_count):
                    self.table_widget_multi_results.insertRow(row_num)
                    for j, value in enumerate(results_df.values.tolist()[0]):
                        if results_df.columns[j] in ["corr_mean", "mmc_mean", "corr_std", "mmc_std", "corr_sharpe", "mmc_sharpe", "corr_max_drawdown", "mmc_max_drawdown"]:
                            # Extract the float number using regular expressions
                            match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
                            if match:
                                float_value = float(match.group())
                                item = QTableWidgetItem(f'{float_value:.6f}')
                                self.table_widget_multi_results.setItem(row_num, j, item)
                            else:
                                # Handle cases where no float number is found
                                item = QTableWidgetItem()
                                self.table_widget_multi_results.setItem(row_num, j, item)
                        else:
                            item = QTableWidgetItem(str(value))
                            self.table_widget_multi_results.setItem(row_num, j, item)
     
                    button = QPushButton("View Model")
                    button.clicked.connect(lambda checked, model=model, selected_feature_sets=selected_feature_sets: self.function_download_Live_predictions_for_a_row(model, selected_feature_sets))
                    self.table_widget_multi_results.setCellWidget(row_num, len(headers) - 1, button)
                    
                self.body_widget.setCurrentIndex(2)
            
        except Exception as e:
            QMessageBox.warning(self, f"Cells not balanced in numbers {hyperparameters_grid}: {e}")
            raise RuntimeError("Results not generated due to entry error")
        
      
    def function_download_Live_predictions_for_a_row(self, model ,selected_feature_sets):
        if model is None or selected_feature_sets is None:
            QMessageBox.warning(None, "Error", "Please train the model first.")
            return None
        
        model_with_predict = ModelWithPredictMethod(model, selected_feature_sets)

        try:
            # Pickle the entire ModelWithPredictMethod class
            model_pickle = cloudpickle.dumps(model_with_predict.predict)
            with open("model.pkl", "wb") as f:
                f.write(model_pickle)
            QMessageBox.information(None, "Success", "Model with predict method pickled successfully.")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Error pickling model: {e}")
               

    def setup_validation_results_table(self, validation):
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
                                                                                          

    def train_lgb_model_with_hyperparameters(self, hyperparameters, selected_feature_sets, train):
        """
        Train a model with specified hyperparameters.

        Args:
            hyperparameters (dict): Dictionary containing hyperparameter values.
            selected_feature_sets (list): List of selected feature sets.
            train (pd.DataFrame): Training data.

        Returns:
            sklearn.base.BaseEstimator: Trained model.
        """
        try:
            # Train the model using the provided features and target data
            model = LGBMRegressor(**hyperparameters)
            model.fit(train[selected_feature_sets], train["target"])
            return model
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error training model: {e}")
            return None
        

        
    def verify_load_validation_file_selected(self):
        """
        Loads validation data.

        Returns:
            str or None: File path of the selected validation file if successful, None otherwise.
        """

   
        # Get the selected validation file and proceed with the rest of the code
        selected_validation_file = self.list_widget_validation_downloaded_datasets.currentItem().text()
        if not selected_validation_file:
            QMessageBox.warning(self, "Selection Required", "Please select a validation file to proceed.")
        return selected_validation_file
    
     
    def load_validation_data(self, selected_feature_sets,selected_validation_file):
        """
        Loads validation data.

        Returns:
            str or None: File path of the selected validation file if successful, None otherwise.
        """
        file_extension_Validation_file = os.path.splitext(selected_validation_file)[1].lower()
        full_file_path_validation_file = os.path.join(self.dynamic_folder_path, selected_validation_file)

        try:
            if file_extension_Validation_file == ".parquet":
                validation = pd.read_parquet(full_file_path_validation_file, columns=["era" ,"data_type", "target"] + selected_feature_sets)
                return validation
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return


    def verify_load_performance_metric_file_selected(self):
        """
        Loads the performance metric file.

        Returns:
            str or None: File path of the selected performance metric file if successful, None otherwise.
        """     
        # Get the selected performance metric file and proceed with the rest of the code
        selected_performance_file = self.list_Widget_meta_model_datasets.currentItem().text()
        if not selected_performance_file:
            QMessageBox.warning(self, "Selection Required", "Please select a Performance file to proceed.")
            return None
        
        return selected_performance_file

    def load_performance_metric_file(self,validation,selected_performance_file):

        file_extension_peformance_file = os.path.splitext(selected_performance_file)[1].lower()
        full_file_path_peformance_file = os.path.join(self.dynamic_folder_path, selected_performance_file)
        try:
            if file_extension_peformance_file == ".parquet":
                validation["meta_model"]= pd.read_parquet(full_file_path_peformance_file)["numerai_meta_model"]
                self.fuction_parquet_data_into_table(validation,self.table_widget_metamodel_performance_file)
                return validation
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return
                
  

    def load_training_data(self, selected_feature_sets):
        """
        Loads training data based on the selected feature sets.

        Args:
            self: Reference to the main class instance to access GUI elements.
            selected_feature_sets (list): Selected feature sets.

        Returns:
            pd.DataFrame or None: Loaded training data if successful, None otherwise.
        """
        selected_training_file = self.list_widget_train_downloaded_datasets.currentItem()
        if not selected_training_file:
            QMessageBox.warning(self, "Selection Required", "Please select a training file to proceed.")
            return None

        selected_training_file = selected_training_file.text()
        file_extension_training_file = os.path.splitext(selected_training_file)[1].lower()
        full_file_path_training_file = os.path.join(self.dynamic_folder_path, selected_training_file)

        try:
            if file_extension_training_file == ".parquet":
                train = pd.read_parquet(full_file_path_training_file, columns=["era", "target"] + selected_feature_sets)
                self.fuction_parquet_data_into_table(train, self.table_widget_train_dataset)
                logging.debug('Loaded training data')
                return train
            else:
                raise ValueError("Unsupported file format.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return None
                
     
    def get_selected_feature_sets(self):
        """
        Retrieves the selected feature sets from the feature file.

        Args:
            self: Reference to the main class instance to access GUI elements.

        Returns:
            list: Selected feature sets if successful.
        """
        if not self.selected_feature_set or not self.selected_feature_file:
            QMessageBox.warning(self, "Selection Required", "Please select a Feature Set and a Feature File to proceed.")
            return None

        file_extension_feature_file = os.path.splitext(self.selected_feature_file)[1].lower()
        full_file_path_feature_file = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

        try:
            if file_extension_feature_file == ".json":
                with open(full_file_path_feature_file, 'r') as file:
                    feature_metadata = json.load(file)
                    selected_feature_sets = feature_metadata["feature_sets"][self.selected_feature_set]
                    return selected_feature_sets
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
            return None
         
    def create_HistGradientBoostingRegressor_hyperparameter_grid(self):
        """
        Creates a hyperparameter grid for LGBMRegressor based on user input.

        Args:
            self: Reference to the main class instance to access GUI elements.

        Returns:
            dict or None: Hyperparameter grid if successful, None otherwise.
        """
        # Get hyperparameters values from GUI inputs
        max_iteration_value=self.max_iter_sci.text()
        learning_rate_value = self.learning_rate_sci.text()
        max_depth_value= self.max_depth_sci.text()
        num_leaves_value=self.max_leaf_nodes_sci.text()
        max_features_value = self.max_features_sci.text()
    
        # Validate hyperparameters
        if not all([learning_rate_value, max_iteration_value, max_depth_value, num_leaves_value, max_features_value]):
            QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
            return None

        # Split user input based on delimiter
        max_iteration_values= re.split(r'\s|[,;]', max_iteration_value)
        learning_rate_values = re.split(r'\s|[,;]', learning_rate_value)
        max_depth_values = re.split(r'\s|[,;]', max_depth_value)
        num_leaves_values = re.split(r'\s|[,;]', num_leaves_value)
        max_features_values = re.split(r'\s|[,;]', max_features_value)

        # Get the lengths of each parameter list
        lengths = {
            'max_iter': len(max_iteration_values),
            'learning_rate': len(learning_rate_values),
            'max_depth': len(max_depth_values),
            'max_leaf_nodes': len(num_leaves_values),
            'max_features': len(max_features_values)
        }


        # Check if all lengths are equal
        if len(set(lengths.values())) != 1:
            print("Number of values for each hyperparameter is not equal.")
            return None

        try:
            # Create hyperparameters grid based on user input
            hyperparameters_grid = {
                'max_iter': [],
                'learning_rate': [],
                'max_depth': [],
                'max_leaf_nodes': [],
                'max_features': []
            }

            # Check type correspondences for each value and add them to the hyperparameters grid
            for value in max_iteration_values:
                try:
                    hyperparameters_grid['max_iter'].append(int(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid integer for 'max_iter'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in learning_rate_values:
                try:
                    hyperparameters_grid['learning_rate'].append(float(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid float for 'learning_rate'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in max_depth_values:
                try:
                    hyperparameters_grid['max_depth'].append(int(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid integer for 'max_depth'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in num_leaves_values:
                try:
                    hyperparameters_grid['max_leaf_nodes'].append(int(eval(value)))
                except (ValueError, TypeError, NameError, SyntaxError):
                    error_message = f"Error: '{value}' is not a valid expression for 'max_leaf_nodes'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in max_features_values:
                try:
                    hyperparameters_grid['max_features'].append(float(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid float for 'max_features'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

        except Exception as e:
            error_message = f"Unexpected error: {e}"
            QMessageBox.warning(self, "Error", error_message)
            return None

    
        self.max_iter_sci.clear()
        self.learning_rate_sci.clear()
        self.max_depth_sci.clear()
        self.max_leaf_nodes_sci.clear()
        self.max_features_sci.clear()
        
        return hyperparameters_grid     
      
                                 
    def create_lgbm_hyperparameter_grid(self):
        """
        Creates a hyperparameter grid for LGBMRegressor based on user input.

        Args:
            self: Reference to the main class instance to access GUI elements.

        Returns:
            dict or None: Hyperparameter grid if successful, None otherwise.
        """
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
            return None
                
        try:
            # Create hyperparameters grid based on user input
            hyperparameters_grid = {
                'n_estimators': [],
                'learning_rate': [],
                'max_depth': [],
                'num_leaves': [],
                'colsample_bytree': []
            }

            # Check type correspondences for each value and add them to the hyperparameters grid
            for value in n_estimators_values:
                try:
                    hyperparameters_grid['n_estimators'].append(int(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid integer for 'n_estimators'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in learning_rate_values:
                try:
                    hyperparameters_grid['learning_rate'].append(float(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid float for 'learning_rate'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in max_depth_values:
                try:
                    hyperparameters_grid['max_depth'].append(int(value))
                except ValueError:
                    error_message = f"Error: '{value}' is not a valid integer for 'max_depth'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in num_leaves_values:
                try:
                    hyperparameters_grid['num_leaves'].append(int(eval(value)))
                except (ValueError, TypeError, NameError, SyntaxError):
                    error_message = f"Error: '{value}' is not a valid expression for 'num_leaves'."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

            for value in colsample_bytree_values:
                if value.strip():  # Check if the value is not an empty string
                    try:
                        hyperparameters_grid['colsample_bytree'].append(float(value))
                    except ValueError:
                        error_message = f"Error: '{value}' is not a valid float for 'colsample_bytree'."
                        QMessageBox.warning(self, "Input Error", error_message)
                        return None
                else:
                    error_message = "Error: Empty string found in 'colsample_bytree_values'. Please provide a valid float value."
                    QMessageBox.warning(self, "Input Error", error_message)
                    return None

        except Exception as e:
            error_message = f"Unexpected error: {e}"
            QMessageBox.warning(self, "Error", error_message)
            return None
            

        self.n_estimators.clear()
        self.learning_rate.clear()
        self.max_depth.clear()
        self.num_leaves.clear()
        self.colsample_bytree.clear()

        return hyperparameters_grid                                            
           

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

class DelimitedValidator(QValidator):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.previous_char_digit = True  # Flag to track if the previous character was a digit

    def validate(self, input_, state):
        """Validates the input string."""

        # Ignore the state (no Locale check)
        # Check if input is empty
        if not input_:
            return QValidator.Acceptable

        # Check for alphabets and whitespace only
        if any(char.isspace() or char.isalpha() for char in input_):
            return QValidator.Invalid

        return QValidator.Acceptable

    def fixup(self, input_):
        """Fixes up the input string if necessary."""

        if not input_:
            return ""

        return input_


# Your Platform class and other classes go here
if __name__ == "__main__":
    app = QApplication(sys.argv)
    initializer = app_Initializer(app)
    initializer.start()

