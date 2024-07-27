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
from datetime import datetime

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution

from PySide6.QtGui import QPixmap, QValidator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QListWidget, QTableWidgetItem, QListWidgetItem, QTableWidget, QSizePolicy, QAbstractItemView, QSizePolicy, QComboBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QFileDialog,
    QLabel, QPushButton, QMessageBox,QStackedWidget,QLayout
)

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)

class Platform(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        #Intiate Empty Variables 
        self.dynamic_folder_path = self.function_get_default_folder_path()
        self.selected_feature_file = None
        self.selected_feature_set= None
        self.num_of_features = None
        self.live_features_stored = None   
        self.trained_models = {}

        #Intiate Layouts
        self.initialize_Core_layouts()
        self.initialize_Column_0_menu_Layout()
        self.initialize_Column_2_Body_model_layout()

        #Set up terminal redirect 
        self.setup_terminal_redirect()

        #Connect List Widget to NumerAi function
        self.function_Download_all_available_datasets_numerAi()
               
    def setup_terminal_redirect(self):
        """Redirect sys.stdout and sys.stderr to the QTextEdit."""
        try:
            self.stdout_redirector = StdoutRedirector(text_widget=self.terminal_widget)
            self.stderr_redirector = StderrRedirector(text_widget=self.terminal_widget)
            sys.stdout = self.stdout_redirector
            sys.stderr = self.stderr_redirector
        except Exception as e:
            self.show_error_message("Error setting up terminal redirect", str(e))       
        
    def initialize_Core_layouts(self) -> None:
        """
        Initialize Layout for Menu and Body Model
        """
        try:
            main_layout = QHBoxLayout()
            self.setLayout(main_layout)

            layout_styles = {
                "Column_0_menu_layout": "#FFFFFF",
                "Column_2_Body_model_layout": "#FFFFFF"   
            }

            for layout_attr_name, background_color in layout_styles.items():
                try:
                    layout_widget = QWidget()  # Create a widget to hold the layout
                    layout_widget.setStyleSheet(f"background-color: {background_color};")
                    layout = QVBoxLayout(layout_widget)
                    setattr(self, layout_attr_name, layout)
                    layout.setContentsMargins(5, 5, 5, 5)
                    main_layout.addWidget(layout_widget)  # Add the layout widget to the main layout
                
                except Exception as e:
                    self.show_error_message(f"Error creating or adding layout {layout_attr_name}", str(e))
        except Exception as e:
            self.show_error_message("Error initializing main layout", str(e))

    def show_error_message(self, title, message):
        """Show an error message dialog."""
        QMessageBox.critical(self, title, message)
             
    def initialize_Column_0_menu_Layout(self)-> None:
        try:    
            column_0_layout_internal = QVBoxLayout()

            button_style = """
                background-color: #001357;
                color: #F9FFFF;
                font-family: Arial;
                font-size: 14px;
                border: 2px solid #000000;
                border-radius: 5px;
                padding: 5px 10px;
                text-align: center;
            """
 
            # Create four buttons forming the Menu
            self.button_train_model_step_1 = self.function_create_button("Data", column_0_layout_internal, self.function_hide_other_layouts, button_style)
            self.button_validate_model_step_2= self.function_create_button("Train", column_0_layout_internal, self.function_hide_other_layouts,button_style)
            self.button_live_model_step_3= self.function_create_button("Outcome", column_0_layout_internal, self.function_hide_other_layouts,button_style)
            self.button_peformance_metrics_step_4 = self.function_create_button("Tables", column_0_layout_internal, self.function_hide_other_layouts,button_style)
 
            # Add a vertical spacer to push buttons to the top
            column_0_layout_internal.addStretch(1)

            # Add internal layout to the main layout
            if hasattr(self, 'Column_0_menu_layout'):
                self.Column_0_menu_layout.addLayout(column_0_layout_internal)
            else:
                raise AttributeError("self.Column_0_menu_layout does not exist")
            
        except Exception as e:
            self.show_error_message("Error initializing Column 0 menu layout", str(e))
             
    def showEvent(self, event):
        super().showEvent(event)
        # Cap the maximum width of the Column_0_menu_layout parent widget to be 30% of the total width
        parent_widget = self.Column_0_menu_layout.parentWidget()
        if parent_widget:
            parent_widget.setMaximumWidth(self.width() * 0.25)
        
    def initialize_Column_2_Body_model_layout(self)-> None:
        """
        Initialize Layout for Body Model
        """
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
        background_style = "background-color: #FFFFFF;"
        header_label_style = "font-size: 20px; color: #001357; font-family: Roboto;"
        body_background_style = "background-color: #FFFFFF;"

        try:
            # Create the header widget
            header_widget = QWidget()
            column_2_layout_internal_Header = QVBoxLayout(header_widget)
            header_widget.setStyleSheet(background_style)
            
            # Create the label for displaying button text
            self.button_label = QLabel("Data")
            self.button_label.setStyleSheet(header_label_style)
            column_2_layout_internal_Header.addWidget(self.button_label)

            # Create the body widget
            self.body_widget = QStackedWidget()
            self.body_widget.setStyleSheet(body_background_style)

            # Create and add different pages to the stacked widget
            self.page_dataset  = self.set_layout_dataset()
            self.page_train  = self.set_layout_to_train()
            self.page_live  = self.set_layout_to_Automate()
            self.page_performance  = self.set_layout_to_peformance()

            self.body_widget.addWidget(self.page_dataset)
            self.body_widget.addWidget(self.page_train)
            self.body_widget.addWidget(self.page_live)
            self.body_widget.addWidget(self.page_performance)

            # Create the end widget
            end_widget = QWidget()
            column_2_layout_internal_End = QVBoxLayout(end_widget)
            end_widget.setStyleSheet(background_style)

            # Create the terminal widget
            self.terminal_widget = QTextEdit()
            self.terminal_widget.setStyleSheet(Text_Edit_Style)
            self.terminal_widget.setReadOnly(True)  # Set read-only property
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
            self.show_error_message("Error initializing Column 2 body model layout", str(e))
            
    def set_layout_dataset(self):
        """
        Set up the layout for the dataset page.
        """
        try:
            # Define styles
            styles = {
                "left_column": {
                    "button": """
                        background-color: #001357;
                        color: #F9FFFF;
                        font-family: Arial;
                        font-size: 14px;
                        border: 2px solid #000000;
                        border-radius: 5px;
                        padding: 5px 10px;
                        text-align: center;
                    """,
                    "label": """
                        background-color: #FFFFFF;  /* Light teal */
                        color: #001357;
                        font-family: Arial;
                        font-size: 12px;
                    """,
                    "list": """
                        background-color: #F0F0F0;  /* Light grey */
                        color: #000000;
                        border: 1px solid #D0D0D0;  /* Grey border */
                    """,
                    "lineedit": """
                        color: #000000;
                        font-size: 12px;
                        font-family: Arial;
                        background-color: #F5F5F5;  /* Light grey background */
                        border: 1px solid #B2B2B2;  /* Light grey border */
                        border-radius: 5px;
                        padding: 5px;
                        selection-background-color: #B3E5FC;  /* Light blue */
                        selection-color: #000000;
                    """
                },
                "right_column": {
                    "button": """
                        background-color: #001357;
                        color: #F9FFFF;
                        font-family: Arial;
                        font-size: 14px;
                        border: 2px solid #000000;
                        border-radius: 5px;
                        padding: 5px 10px;
                        text-align: center;
                    """,
                    "label": """
                        background-color: #FFFFFF;  /* Light teal */
                        color: #001357; 
                        font-family: Arial;
                        font-size: 12px;
                    """,
                    "list": """
                        background-color: #F0F0F0;  /* Light grey */
                        color: #000000;
                        border: 1px solid #D0D0D0;  /* Grey border */
                    """,
                    "lineedit": """
                        color: #000000;
                        font-size: 12px;
                        font-family: Arial;
                        background-color: #FAFAFA;  /* Very light grey */
                        border: 1px solid #D0D0D0;  /* Grey border */
                        border-radius: 5px;
                        padding: 5px;
                        selection-background-color: #B3E5FC;  /* Light blue */
                        selection-color: #000000;
                    """
                }
            }

            page_widget = QWidget()
            layout = QHBoxLayout(page_widget)

            # Left column
            left_column_layout = QVBoxLayout()

            self.function_create_label("NUMERAI Available Datasets", left_column_layout,styles["left_column"]["label"])
            self.list_widget_all_datasets = QListWidget()
            self.list_widget_all_datasets.setStyleSheet(styles["left_column"]["list"])
            self.list_widget_all_datasets.setSelectionMode(QAbstractItemView.MultiSelection)
            left_column_layout.addWidget(self.list_widget_all_datasets)

            # Set a default folder path based on the operating system
            self.folder_path_edit = QLineEdit(self.dynamic_folder_path, self)
            self.folder_path_edit.setStyleSheet(styles["left_column"]["lineedit"])
            left_column_layout.addWidget(self.folder_path_edit)
            
            self.browse_button = self.function_create_button("Folder Browse", left_column_layout,self.button_function_browse_folder, styles["left_column"]["button"])

            self.button_download_data_set_selected = self.function_create_button("Download Selected Datasets", left_column_layout, self.button_function_download_selected_dataset,styles["left_column"]["button"])
            
            self.function_create_label("Downloaded Datasets", left_column_layout,styles["left_column"]["label"])

            self.list_Widget_Availabile_Downloaded_Datasets = QListWidget()
            self.list_Widget_Availabile_Downloaded_Datasets.setStyleSheet(styles["left_column"]["list"])
            
            
            left_column_layout.addWidget(self.list_Widget_Availabile_Downloaded_Datasets)

            layout.addLayout(left_column_layout)
            
            # Right column
            right_column_layout = QVBoxLayout()   

            # Section for feature List  - Column Right  
            self.function_create_label("Features List - Select One", right_column_layout,styles["right_column"]["label"])
            self.list_Widget_Features_Downloaded_Datasets= QListWidget()
            self.list_Widget_Features_Downloaded_Datasets.setStyleSheet(styles["right_column"]["list"])
            right_column_layout.addWidget(self.list_Widget_Features_Downloaded_Datasets)
            self.list_Widget_Features_Downloaded_Datasets.itemClicked.connect(self.function_display_feature_list)  

            # Section for feature List Content (auto popu;ated list)  - Column Right   
            self.function_create_label("Features List Contents - Select One", right_column_layout,styles["right_column"]["label"])
            self.list_widget_features_content= QListWidget()
            self.list_widget_features_content.setStyleSheet(styles["right_column"]["list"])
            right_column_layout.addWidget(self.list_widget_features_content)
            self.list_widget_features_content.itemClicked.connect(self.function_handle_feature_list_change)

            # Section for Training Dataset List  - Column Right       
            self.function_create_label("Training Dataset List - Select One", right_column_layout,styles["right_column"]["label"])
            self.list_widget_train_downloaded_datasets = QListWidget()
            self.list_widget_train_downloaded_datasets.setStyleSheet(styles["right_column"]["list"])
            right_column_layout.addWidget(self.list_widget_train_downloaded_datasets)
            
            # Creation of a Validation Dataset List - Column Right   
            self.function_create_label("Validation list - Select One", right_column_layout,styles["right_column"]["label"])
            self.list_widget_validation_downloaded_datasets= QListWidget()     
            self.list_widget_validation_downloaded_datasets.setStyleSheet(styles["right_column"]["list"])   
            right_column_layout.addWidget(self.list_widget_validation_downloaded_datasets)
            
            # Creation of a Performance Metric Dataset List - Column Right  
            self.function_create_label("Performance metrics List - Select One", right_column_layout,styles["right_column"]["label"])
            self.list_Widget_meta_model_datasets = QListWidget()
            self.list_Widget_meta_model_datasets.setStyleSheet(styles["right_column"]["list"])
            right_column_layout.addWidget(self.list_Widget_meta_model_datasets)
            
            #Meta Data - Empty Parameters - Column Right 
            self.metadata_widget = QWidget()  
            metadata_layout = QVBoxLayout()    
            self.metadata_widget.setLayout(metadata_layout)
            right_column_layout.addWidget(self.metadata_widget)
                
            layout.addLayout(right_column_layout)       

            self.function_update_downloaded_datasets_list(self.dynamic_folder_path) 
               
            return page_widget
        except Exception as e:
            self.show_error_message("Error setting up the dataset layout", str(e)) 

    def set_layout_to_train(self):
        button_style = """
            background-color: #001357;
            color: #F9FFFF;
            font-family: Arial;
            font-size: 14px;
            border: 2px solid #000000;
            border-radius: 5px;
            padding: 5px 10px;
            text-align: center;
        """
        label_style = """
            background-color: #FFFFFF;  /* Light teal */
            color: #001357; 
            font-family: Arial;
            font-size: 12px;
        """
        combobox_style = """
            color: #001357;
            font-size: 12px;
            font-family: Arial;
            background-color: #F9FFFF;
            border: 1px solid #CCCCCC;
        """

        try:
            page_widget = QWidget()
            layout = QHBoxLayout(page_widget)
            
            # Left column layout
            left_column_layout = QVBoxLayout()

            # ComboBox for training methods
            self.function_create_label("Training Module - Select One", left_column_layout, label_style)
            self.training_method_combo = QComboBox()
            self.training_method_combo.setStyleSheet(combobox_style)
            self.training_method_combo.addItems(["LGBMRegressor", "HistGradientBoostingRegressor"])
            self.training_method_combo.currentIndexChanged.connect(self.function_training_method_changed)
            left_column_layout.addWidget(self.training_method_combo)
            
            # Create container for LGBMRegressor hyperparameters
            self.container_LGM = QWidget()
            self.hyperparameters_layout_LGM = QVBoxLayout(self.container_LGM)
            self.n_estimators = self.function_create_labeled_lineedit(
                "Number of boosted trees to fit",
                self.hyperparameters_layout_LGM,
                "Please input Integer values like 2000. For multiple values, separate them with commas like 2000, 3000",
                DelimitedValidator
            )
            self.learning_rate = self.function_create_labeled_lineedit(
                "Boosting learning rate",
                self.hyperparameters_layout_LGM,
                "Please input float values like 0.01. For multiple values, separate them with commas like 0.01, 0.02",
                DelimitedValidator
            )
            self.max_depth = self.function_create_labeled_lineedit(
                "Maximum tree depth for base learners",
                self.hyperparameters_layout_LGM,
                "Please input Integer values like 5. For multiple values, separate them with commas like 5, 6",
                DelimitedValidator
            )
            self.num_leaves = self.function_create_labeled_lineedit(
                "Maximum tree leaves for base learners",
                self.hyperparameters_layout_LGM,
                "Please input Integer values like 30. For multiple values, separate them with commas like 30, 31",
                DelimitedValidator
            )
            self.colsample_bytree = self.function_create_labeled_lineedit(
                "Subsample ratio of columns when constructing each tree",
                self.hyperparameters_layout_LGM,
                "Please input float values like 0.1. For multiple values, separate them with commas like 0.1, 0.25",
                DelimitedValidator
            )
            
            # Initially hide the LGBM container
            self.container_LGM.setVisible(False)
            left_column_layout.addWidget(self.container_LGM)
            
            # Create container for HistGradientBoostingRegressor hyperparameters
            self.container_HGBR = QWidget()
            self.hist_gradient_boosting_layout = QVBoxLayout(self.container_HGBR)
            self.learning_rate_sci = self.function_create_labeled_lineedit(
                "Learning Rate",
                self.hist_gradient_boosting_layout,
                "Please input float values like 0.01. For multiple values, separate them with commas like 0.01, 0.02",
                DelimitedValidator
            )
            self.max_iter_sci = self.function_create_labeled_lineedit(
                "Max Number of Trees",
                self.hist_gradient_boosting_layout,
                "Please input Integer values like 2000. For multiple values, separate them with commas like 2000, 3000",
                DelimitedValidator
            )
            self.max_leaf_nodes_sci = self.function_create_labeled_lineedit(
                "Max Num of leaves per Tree",
                self.hist_gradient_boosting_layout,
                "Please input Integer values like 31. For multiple values, separate them with commas like 30, 31",
                DelimitedValidator
            )
            self.max_depth_sci = self.function_create_labeled_lineedit(
                "Max Depth of Tree",
                self.hist_gradient_boosting_layout,
                "Please input Integer values like 5. For multiple values, separate them with commas like 5, 6",
                DelimitedValidator
            )
            self.max_features_sci = self.function_create_labeled_lineedit(
                "Proportion of random chosen features in each node split",
                self.hist_gradient_boosting_layout,
                "Please input float values like 0.1. For multiple values, separate them with commas like 0.1, 0.2",
                DelimitedValidator
            )
            
            # Initially hide the HistGradientBoostingRegressor container
            self.container_HGBR.setVisible(False)
            left_column_layout.addWidget(self.container_HGBR)
            
            # Create download button
            self.button_download_data_set_selected = self.function_create_button(
                "Train Multi Models", left_column_layout, self.function_Multiple_Train_Buttons, button_style
            )
            
            left_column_layout.setAlignment(Qt.AlignTop)
            layout.addLayout(left_column_layout)
            
            # Right column layout
            right_column_layout = QVBoxLayout()

            # Image display
            image_label = QLabel()
            pixmap = QPixmap("C:/Users/aat2g/OneDrive/Documents/13. Personal Project/7. NumerAi Test/Numerai-modelling-app/learning.jpg")
            image_label.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            right_column_layout.addWidget(image_label, alignment=Qt.AlignCenter)

            layout.addLayout(right_column_layout)

            # Set default selection and show initial widgets
            self.training_method_combo.setCurrentIndex(0)
            self.function_training_method_changed(0)  # Call the method to show/hide widgets based on default selection

            return page_widget

        except Exception as e:
            print(f"An error occurred while setting up the layout: {e}")

    def set_layout_to_peformance(self):
        """
        Set up the layout for performance metrics with improved UI and error handling.
        """
        try:
            # Define styles
            styles = {
                "label": """
                    background-color: #FFFFFF;  /* Light teal */
                    color: #001357; 
                    font-family: Arial;
                    font-size: 12px;
                """,
                "table": """
                    color: #000000;
                    font-size: 9px;
                    font-family: Arial;
                    background-color: #FFFFFF;
                    border: 1px solid #CCCCCC;
                    border-radius: 2px;
                    selection-background-color: #BFE6FF;
                    selection-color: #000000;
                """
            } 
           
            page_widget = QWidget()
            layout = QHBoxLayout(page_widget)

            # Left column layout
            left_column_layout = QVBoxLayout()
            self.function_add_labeled_table("First 50 Rows Training", left_column_layout, styles["label"], styles["table"], "table_widget_train_dataset")
            self.function_add_labeled_table("First 50 Rows Validation Table", left_column_layout, styles["label"], styles["table"], "table_widget_validation_dataset")
            layout.addLayout(left_column_layout)
            
            # Right column layout
            right_column_layout = QVBoxLayout()
            self.function_add_labeled_table("Validation Results Table", right_column_layout, styles["label"], styles["table"], "table_widget_validation_results")
            self.function_add_labeled_table("Meta Model Performance Table", right_column_layout, styles["label"], styles["table"], "table_widget_metamodel_performance_file")
            layout.addLayout(right_column_layout)

            return page_widget
        except Exception as e:
            self.show_error_message("Error setting up the performance layout", str(e))
      
    def set_layout_to_Automate(self):
        """
        Set up the layout for automation results with improved UI and error handling.
        """
        try:
            # Define styles
            styles = {
                "table": """
                    color: #000000;
                    font-size: 9px;
                    font-family: Arial;
                    background-color: #FFFFFF;
                    border: 1px solid #CCCCCC;
                    border-radius: 2px;
                    selection-background-color: #BFE6FF;
                    selection-color: #000000;
                """
            }           
 
            page_widget = QWidget()
            layout = QVBoxLayout(page_widget)

            # Add table widget
            self.table_widget_multi_results = QTableWidget()
            self.table_widget_multi_results.setStyleSheet(styles["table"])
            layout.addWidget(self.table_widget_multi_results)
            
            # Create a horizontal layout for the graphs
            graph_layout = QHBoxLayout()

            # Add first graph widget and its toolbar
            self.graph_widget_1, self.graph_widget_1_toolbar = self.function_create_graph_widget()
            graph_layout.addWidget(self.graph_widget_1)
            graph_layout.addWidget(self.graph_widget_1_toolbar)

            # Add second graph widget and its toolbar
            self.graph_widget_2, self.graph_widget_2_toolbar = self.function_create_graph_widget()
            graph_layout.addWidget(self.graph_widget_2)
            graph_layout.addWidget(self.graph_widget_2_toolbar)

            # Add the graph layout to the main layout
            layout.addLayout(graph_layout)

            # Connect table selection signal to update graphs
            self.table_widget_multi_results.itemSelectionChanged.connect(self.function_update_graphs)
        
            return page_widget
        except Exception as e:
            self.show_error_message("Error setting up the automation layout", str(e))
  
    def function_hide_other_layouts(self):
        """
        Hide all the layouts related to different steps except the one associated with the clicked button.
        """
        sender_button = self.sender()

        if not sender_button:
            QMessageBox.warning(self, "Error", "No button sender detected.")
            return
    
        sender_name = sender_button.text()

            # Define layout indices
        layout_indices = {
            "Data": 0,
            "Train": 1,
            "Tables": 3,
            "Outcome": 2
            }

        target_index = layout_indices.get(sender_name)
        if target_index is None:
            QMessageBox.warning(self, "Error", "Unknown button clicked.")
            return

        # Check if the current index is not already set to the target index
        if self.body_widget.currentIndex() != target_index:
            if sender_name == "Train":
                # Validate required lists for the "Train" button
                missing_lists = []
                list_validations = {
                    self.list_Widget_Features_Downloaded_Datasets: "Features List",
                    self.list_widget_validation_downloaded_datasets: "Validation Dataset List",
                    self.list_widget_train_downloaded_datasets: "Training Dataset List",
                    self.list_Widget_meta_model_datasets: "Performance Metrics List",
                    self.list_widget_features_content: "Features List Contents"
                }
                
                for list_widget, list_name in list_validations.items():
                    if not list_widget.selectedItems():
                        missing_lists.append(list_name)
                
                if missing_lists:
                    QMessageBox.warning(
                        self, "Missing Selection",
                        f"Please download and select items from the following lists: {', '.join(missing_lists)}"
                    )
                    return
            
            # Set button label and change the current index
            self.button_label.setText(sender_name)
            self.body_widget.setCurrentIndex(target_index)

    def function_create_graph_widget(self):
        graph_widget = FigureCanvas(Figure(figsize=(7, 5)))  # Set initial size of the figure
        toolbar = NavigationToolbar(graph_widget, self)
        graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return graph_widget, toolbar

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

    def function_create_labeled_lineedit(self, label_text, layout, placeholder_text, validator_class=None):
        """
        Creates a labeled QLineEdit with a custom validator and adds it to the layout.
        """


        # Create and style the label
        label = QLabel(label_text)
        label.setStyleSheet("""
            background-color: #FFFFFF;  /* Light teal */
            color: #001357; 
            font-family: Arial;
            font-size: 12px;
        """)
        layout.addWidget(label)

        # Create and style the QLineEdit
        lineedit = QLineEdit()
        lineedit.setStyleSheet("""
            color: #000000;
            font-size: 10px;
            font-family: Arial;
            background-color: #FFFFFF;
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            selection-background-color: #BFE6FF;
            selection-color: #000000;
        """)
        lineedit.setPlaceholderText(placeholder_text)
        if validator_class:
            validator = validator_class()
            lineedit.setValidator(validator)
        
        layout.addWidget(lineedit)
        return lineedit
    
    def function_add_labeled_table(self, label_text, layout, label_style, table_style, table_name):
        """
        Helper function to add a labeled table to a layout and store it as an instance variable.
        
        :param label_text: The text for the label above the table.
        :param layout: The layout to which the label and table will be added.
        :param label_style: The stylesheet for the label.
        :param table_style: The stylesheet for the table.
        :param table_name: The name of the instance variable to store the table widget.
        """
        # Create and add the label
        self.function_create_label(label_text, layout, label_style)
        
        # Create the table widget and set its style
        table_widget = QTableWidget()
        table_widget.setStyleSheet(table_style)
        
        # Add the table widget to the layout
        layout.addWidget(table_widget)
        
        # Store the table widget as an instance variable
        setattr(self, table_name, table_widget)

    def function_create_button(self, text: str, parent_layout, clicked_handler=None, style = None) -> None:
        button = QPushButton(text)
        button.setFixedHeight(30) 
        if style:
            button.setStyleSheet(style)
        if clicked_handler:
            button.clicked.connect(clicked_handler)
             
        parent_layout.addWidget(button)
         
    def function_handle_feature_list_change(self, selected_item):
        """
        Handles changes in the selected feature list, updating relevant attributes and UI elements.

        Args:
            selected_item (QListWidgetItem): The currently selected item from the feature list.
        """
        
        label_style = """
            background-color: #FFFFFF;
            color: #001357;
            font-family: Arial;
            font-size: 12px;
        """
        full_file_path_feature_file = None

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
            except FileNotFoundError:
                print(f"File not found: {full_file_path_feature_file}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {full_file_path_feature_file}")
            except KeyError:
                print(f"Feature set '{self.selected_feature_set}' not found in the file.")
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
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

        # Update the UI with new information
        self.function_create_label(f"Feature List File: {self.selected_feature_file if self.selected_feature_file else 'N/A'}", layout, label_style)
        self.function_create_label(f"Feature List Selected: {self.selected_feature_set if self.selected_feature_set else 'N/A'}", layout, label_style)
        self.function_create_label(f"Number of Features: {self.num_of_features if self.num_of_features > 0 else 'N/A'}", layout, label_style)
        self.function_create_label(f"Folder Path: {full_file_path_feature_file if full_file_path_feature_file else 'N/A'}", layout, label_style)        
                
    def function_get_default_folder_path(self):
        # Default folder path based on the OS
        default_folder = os.path.join(os.path.expanduser("~"), "Downloads")

        # Create a sandbox folder within the default Downloads folder
        sandbox_folder = os.path.join(default_folder, "sandbox")

        # Attempt to create the sandbox folder if it doesn't exist
        try:
            if not os.path.exists(sandbox_folder):
                os.makedirs(sandbox_folder)
        except Exception as e:
            # Inform the user if the folder cannot be created
            QMessageBox.warning(
                self,
                "Folder Creation Not possible due to User Setup",
                "The application could not create the default folder for storing files. "
                "Please use the browse button to select a folder manually."
            )
            # Fallback to default folder path
            self.dynamic_folder_path = default_folder
            return self.dynamic_folder_path

        self.dynamic_folder_path = sandbox_folder
        return self.dynamic_folder_path

    def button_function_browse_folder(self):
        # Open a dialog to allow the user to select a folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder_path:
            # Prompt the user to confirm the folder path change
            reply = QMessageBox.question(
                self,
                "Confirm Folder Path",
                f"Do you want to change the folder path to:\n{folder_path}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.folder_path_edit.setText(folder_path)
                self.function_update_downloaded_datasets_list(folder_path)
                self.dynamic_folder_path = folder_path
            else:
                # Optionally, you can add a message indicating no changes were made
                QMessageBox.information(self, "No Change", "The folder path was not changed.")

        else:
            # Inform the user if no folder was selected
            QMessageBox.warning(self, "No Folder Selected", "No folder was selected. Please try again.")
        
    def function_training_method_changed(self, index):
        """
        Show or hide hyperparameter sections based on the selected training method.
        """
        try:
            if index == 0:  # LGBMRegressor
                self.container_LGM.setVisible(True)
                self.container_HGBR.setVisible(False)
            elif index == 1:  # HistGradientBoostingRegressor
                self.container_LGM.setVisible(False)
                self.container_HGBR.setVisible(True)
        except Exception as e:
            print(f"An error occurred while changing training method: {e}")
                        
    def function_Download_all_available_datasets_numerAi(self) -> None:
        # Clear existing items from the list widget
        self.list_widget_all_datasets.clear()

        # Initialize NumerAPI - the official Python API client for Numerai
        napi = NumerAPI()

        # Retrieve the current date for logging purposes
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # List the datasets and available versions
            all_datasets = napi.list_datasets()
            
            if all_datasets:
                # Log the number of files and list of files to the terminal
                print(f"[{current_date}] Datasets found: {', '.join(all_datasets)}")
                print(f"[{current_date}] Number of datasets found: {len(all_datasets)}")
                
                # Add datasets to the list widget
                self.list_widget_all_datasets.addItems(all_datasets)
            else:
                # Inform the user if no datasets were found
                QMessageBox.warning(
                    self,
                    "No Datasets Found",
                    "No datasets were found from Numerai. Please check your API connection or try again later."
                )
                print(f"[{current_date}] No datasets found.")
                
        except Exception as e:
            # Inform the user of the error and log it to the terminal
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while retrieving datasets from Numerai:\n{e}"
            )
            print(f"[{current_date}] Error retrieving datasets from Numerai: {e}")
                                        
    def function_update_downloaded_datasets_table(self, dataset=None) -> None:
        """
        Updates the list of downloaded datasets.
        """
        try:       
            # Initialize empty sets for existing items
            existing_items_train = set()
            existing_items_validation = set()
            existing_items_features = set()
            existing_items_downloaded = set()
            existing_meta_model_downloaded = set()

            # Check if each list widget is defined before attempting to collect existing items
            widgets = {
                'list_widget_train_downloaded_datasets': existing_items_train,
                'list_widget_validation_downloaded_datasets': existing_items_validation,
                'list_Widget_Features_Downloaded_Datasets': existing_items_features,
                'list_Widget_Availabile_Downloaded_Datasets': existing_items_downloaded,
                'list_Widget_meta_model_datasets': existing_meta_model_downloaded
            }

            for widget_name, item_set in widgets.items():
                if hasattr(self, widget_name):
                    item_set.update(
                        self.helper_function_find_widget_items(getattr(self, widget_name))
                    )
                else:
                    print(f"Warning: {widget_name} is not defined.")
                    return

        except AttributeError as e:
            print(f"Error accessing widgets: {e}")
            return
        
        # Update list widgets with new items
        for item in existing_items_downloaded:
            item_lower = item.lower()
            if "feature" in item_lower and item not in existing_items_features:
                self.list_Widget_Features_Downloaded_Datasets.addItem(item)

            if "validation" in item_lower and item not in existing_items_validation:
                self.list_widget_validation_downloaded_datasets.addItem(item)

            if "train" in item_lower and item not in existing_items_train:
                self.list_widget_train_downloaded_datasets.addItem(item)

            if "meta_model" in item_lower and item not in existing_meta_model_downloaded:
                self.list_Widget_meta_model_datasets.addItem(item)

        # Handle newly selected dataset
        if dataset is not None:
            dataset_text = dataset.text()
            dataset_lower = dataset_text.lower()

            if dataset_text not in existing_items_downloaded:
                self.list_Widget_Availabile_Downloaded_Datasets.addItem(dataset_text)

            if "feature" in dataset_lower and dataset_text not in existing_items_features:
                self.list_Widget_Features_Downloaded_Datasets.addItem(dataset_text)
                    
            if "validation" in dataset_lower and dataset_text not in existing_items_validation:
                self.list_widget_validation_downloaded_datasets.addItem(dataset_text)
                    
            if "train" in dataset_lower and dataset_text not in existing_items_train:
                self.list_widget_train_downloaded_datasets.addItem(dataset_text)

            if "meta_model" in dataset_lower and dataset_text not in existing_meta_model_downloaded:
                self.list_Widget_meta_model_datasets.addItem(dataset_text)

    def helper_function_find_widget_items(self, widget):
        """
        Helper function to get the text of all items in a list widget.
        """
        return {widget.item(i).text() for i in range(widget.count())}
                
    def function_update_downloaded_datasets_list(self, folder_path) -> None:
        """
        Updates the list of downloaded datasets.
        """
        # Clear the lists if they exist
        for widget_name in [
            'list_Widget_Availabile_Downloaded_Datasets',
            'list_Widget_Features_Downloaded_Datasets',
            'list_widget_validation_downloaded_datasets',
            'list_widget_train_downloaded_datasets',
            'list_Widget_meta_model_datasets',
            'list_widget_features_content'
        ]:
            if hasattr(self, widget_name):
                getattr(self, widget_name).clear()

        # Reset feature list handler
        self.function_handle_feature_list_change(None)

        try:
            # Iterate over the directory and files
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.csv', '.parquet', '.json', '.xlsx', '.db', '.sqlite', '.sqlite3')):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, folder_path).replace('\\', '/')
                        self.list_Widget_Availabile_Downloaded_Datasets.addItem(relative_path)
            
            # Update the downloaded datasets table
            self.function_update_downloaded_datasets_table()
        
        except Exception as e:
            print(f"Error updating downloaded datasets list: {e}")
            # Inform the user via a dialog box
            QMessageBox.critical(self, "Error", f"Failed to update downloaded datasets list: {e}")
                         
    def button_function_download_selected_dataset(self)->None:
        # Initialize NumerAPI - the official Python API client for Numerai
        napi = NumerAPI()
        List_selected_Datasets = self.list_widget_all_datasets.selectedItems()

        for dataset in List_selected_Datasets:
            dataset_text = dataset.text()  # Get the text of the selected dataset
            print(f"Preparing to download dataset: {dataset_text}")

            # Define the file path where the dataset will be saved
            if self.dynamic_folder_path:
                file_path = os.path.join(self.dynamic_folder_path, dataset_text)
                file_path = file_path.replace('\\', '/')

                # Check if the file already exists
                if os.path.exists(file_path):
                    # Prompt the user to confirm replacement
                    reply = QMessageBox.question(
                        self,
                        'File Already Exists',
                        f"The file '{dataset_text}' already exists. Do you want to replace it? It's recommended to always use the latest revisions, especially for validation datasets which are updated weekly.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        print(f"User chose not to replace: {dataset_text}")
                        continue

                # Download dataset and save it in the sandbox directory
                try:
                    napi.download_dataset(dataset_text, dest_path=file_path)
                    print(f"Successfully downloaded dataset: {dataset_text}")
                    # Update the downloaded datasets table
                    self.function_update_downloaded_datasets_table(dataset)
                except Exception as e:
                    print(f"Error downloading dataset: {e}")
                     
    def function_display_feature_list(self, item) ->None: 
        """
        Displays the content of the selected feature file in the list widget.
        Supports JSON files with feature sets.
        """   
        self.selected_feature_file = item.text()
        file_extension = os.path.splitext(self.selected_feature_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

        # Check if the file exists before attempting to open it
        if not os.path.exists(full_file_path):
            QMessageBox.warning(self, "Error", f"The file '{self.selected_feature_file}' does not exist.")
            return
    
        try:
            if file_extension == ".json":
                with open(full_file_path, 'r') as file:
                    feature_metadata = json.load(file)

                # Clear existing contents of the list widget
                self.list_widget_features_content.clear()

                # Get feature sets data
                feature_sets_data = feature_metadata.get('feature_sets', {})

                # Populate list widget with feature sets keys
                if not feature_sets_data:
                    QMessageBox.information(self, "Info", "No feature sets found in the file.")
                else:
                    for key in feature_sets_data.keys():
                        item = QListWidgetItem(str(key))
                        self.list_widget_features_content.addItem(item)

                # Populate list widget with feature sets keys
                for key in feature_sets_data.keys():
                    item = QListWidgetItem(str(key))
                    self.list_widget_features_content.addItem(item)
            else:
                raise ValueError("Unsupported file format. Only JSON files are supported.")

        except json.JSONDecodeError as jde:
            QMessageBox.warning(self, "Error", f"Error decoding JSON file: {jde}")

        except ValueError as ve:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {ve}")
            
        except IOError as ioe:
            QMessageBox.warning(self, "Error", f"Error reading file: {ioe}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An unexpected error occurred: {e}")

    def fuction_parquet_data_into_table(self, df, table_widget)->None:
        # Clear existing contents of the table widget
        self.helper_function_clear_table_widget(table_widget)
        
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
           
    def helper_function_clear_table_widget(self, table_widget)->None:
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)    
             
    def function_Multiple_Train_Buttons(self) -> None: 
        """
        Trains multiple models based on selected hyperparameters and feature sets,
        and displays the results in a table widget.
        """
        # Retrieve selected feature sets
        selected_feature_sets = self.get_selected_feature_sets()
        if selected_feature_sets is None:
            QMessageBox.warning(self, "Selection Error", "No feature sets selected.")
            return
        
        # Load training data
        train = self.load_training_data(selected_feature_sets)
        if train is None:
            QMessageBox.warning(self, "Data Error", "Failed to load training data.")
            return
        
        # Load performance metric file
        validation_file_path = self.verify_load_validation_file_selected()
        if validation_file_path is None:
            QMessageBox.warning(self, "File Error", "Validation file not selected or invalid.")
            return
        
        Performance_validation_file_path = self.verify_load_performance_metric_file_selected()
        if Performance_validation_file_path is None:
            QMessageBox.warning(self, "File Error", "Performance metric file not selected or invalid.")
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
        
        # Dictionary to store trained models for the current round
        trained_models_table_this_round_only = {} 
        
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
     
                    button = QPushButton("Download Model")
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

        # Construct the file path using self.dynamic_folder_path with a timestamp
        if self.dynamic_folder_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"model_{timestamp}.pkl"
            file_path = os.path.join(self.dynamic_folder_path, file_name)
        else:
            QMessageBox.warning(None, "Error", "Dynamic folder path is not set.")
            return None

        try:
            # Pickle the entire ModelWithPredictMethod class
            model_pickle = cloudpickle.dumps(model_with_predict.predict)
            with open(file_path, "wb") as f:
                f.write(model_pickle)
            QMessageBox.information(None, "Success", f"Model with predict method pickled successfully to {file_path}.")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Error pickling model: {e}")

    def setup_validation_results_table(self, validation):
        self.helper_function_clear_table_widget(self.table_widget_validation_results)
        
        # Set the column count and headers
        num_columns = 3  # era, prediction, target
        self.table_widget_validation_results.setColumnCount(num_columns + 1)  # Add 1 for the index column
        headers = ["Index", "Era", "Prediction", "Target"]
        self.table_widget_validation_results.setHorizontalHeaderLabels(headers)

        # Check if the validation DataFrame has fewer than 100 rows
        num_rows = len(validation)
        if num_rows <= 50:
            validation_subset = validation
        else:
            validation_subset = pd.concat([validation.head(50), validation.tail(50)])

        # Ensure the required columns exist
        required_columns = ["era", "prediction", "target"]
        if not all(col in validation_subset.columns for col in required_columns):
            QMessageBox.warning(self, "Error", "Validation DataFrame missing required columns.")
            return
        
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
        file_extension = os.path.splitext(selected_validation_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, selected_validation_file)

        try:
            if file_extension == ".parquet":
                # Ensure selected_feature_sets is a list of column names
                if not isinstance(selected_feature_sets, list):
                    raise ValueError("selected_feature_sets should be a list of column names.")

                # Attempt to read the parquet file
                validation = pd.read_parquet(full_file_path, columns=["era", "data_type", "target"] + selected_feature_sets)
                return validation
            else:
                raise ValueError("Unsupported file format.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", f"File not found: {full_file_path}")
        except pd.errors.EmptyDataError:
            QMessageBox.warning(self, "Error", "No data found in the file.")
        except ValueError as ve:
            QMessageBox.warning(self, "Error", f"ValueError: {ve}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")
        
        return None

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

        file_extension = os.path.splitext(selected_performance_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, selected_performance_file)
        try:
            if file_extension == ".parquet":
                # Read the meta_model column from the parquet file
                meta_model_data = pd.read_parquet(full_file_path)["numerai_meta_model"]
                
                # Ensure that the 'meta_model' column is not already present
                if 'meta_model' in validation.columns:
                    QMessageBox.warning(self, "Warning", "Meta model column already exists in the validation data.")
                    return None
                
                # Add the meta_model column to the validation DataFrame
                validation["meta_model"] = meta_model_data
                
                # Update the table with performance metrics
                self.fuction_parquet_data_into_table(validation, self.table_widget_metamodel_performance_file)
                return validation
            else:
                raise ValueError("Unsupported file format.")
            
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", f"File not found: {full_file_path}")
        except pd.errors.EmptyDataError:
            QMessageBox.warning(self, "Error", "No data found in the file.")
        except ValueError as ve:
            QMessageBox.warning(self, "Error", f"ValueError: {ve}")
        except KeyError:
            QMessageBox.warning(self, "Error", "Expected column 'numerai_meta_model' not found in the file.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading performance metric file: {e}")
        
        return None
                  
    def load_training_data(self, selected_feature_sets):
        """
        Loads training data based on the selected feature sets.

        Args:
            selected_feature_sets (list): Selected feature sets.

        Returns:
            pd.DataFrame or None: Loaded training data if successful, None otherwise.
        """
        # Get the currently selected training file
        selected_training_file_item = self.list_widget_train_downloaded_datasets.currentItem()
        if not selected_training_file_item:
            QMessageBox.warning(self, "Selection Required", "Please select a training file to proceed.")
            return None
        
        selected_training_file = selected_training_file_item.text()
        file_extension = os.path.splitext(selected_training_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, selected_training_file)

        try:
            if file_extension == ".parquet":
                # Load the training data from the parquet file
                train = pd.read_parquet(full_file_path, columns=["era", "target"] + selected_feature_sets)
                self.fuction_parquet_data_into_table(train, self.table_widget_train_dataset)
                logging.debug('Loaded training data successfully')
                return train
            else:
                raise ValueError("Unsupported file format. Only .parquet files are supported.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", f"File not found: {full_file_path}")
        except pd.errors.EmptyDataError:
            QMessageBox.warning(self, "Error", "No data found in the file.")
        except ValueError as ve:
            QMessageBox.warning(self, "Error", f"ValueError: {ve}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")

        return None
                  
    def get_selected_feature_sets(self):
        """
        Retrieves the selected feature sets from the feature file.

        Returns:
            list or None: Selected feature sets if successful, None otherwise.
        """
        if not self.selected_feature_set or not self.selected_feature_file:
            QMessageBox.warning(self, "Selection Required", "Please select a Feature Set and a Feature File to proceed.")
            return None

        file_extension = os.path.splitext(self.selected_feature_file)[1].lower()
        full_file_path = os.path.join(self.dynamic_folder_path, self.selected_feature_file)

        try:
            if file_extension == ".json":
                with open(full_file_path, 'r') as file:
                    feature_metadata = json.load(file)
                    selected_feature_sets = feature_metadata["feature_sets"].get(self.selected_feature_set, None)
                    if selected_feature_sets is None:
                        QMessageBox.warning(self, "Error", f"Feature set '{self.selected_feature_set}' not found in the file.")
                    return selected_feature_sets
            else:
                raise ValueError("Unsupported file format. Only .json files are supported.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", f"File not found: {full_file_path}")
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Error", "Error decoding JSON file.")
        except ValueError as ve:
            QMessageBox.warning(self, "Error", f"ValueError: {ve}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading dataset: {e}")

        return None
         
    def create_HistGradientBoostingRegressor_hyperparameter_grid(self):
        """
        Creates a hyperparameter grid for HistGradientBoostingRegressor based on user input.

        Returns:
            dict or None: Hyperparameter grid if successful, None otherwise.
        """
        # Retrieve hyperparameters values from GUI inputs
        try:
            max_iter_values = re.split(r'\s|[,;]', self.max_iter_sci.text())
            learning_rate_values = re.split(r'\s|[,;]', self.learning_rate_sci.text())
            max_depth_values = re.split(r'\s|[,;]', self.max_depth_sci.text())
            num_leaves_values = re.split(r'\s|[,;]', self.max_leaf_nodes_sci.text())
            max_features_values = re.split(r'\s|[,;]', self.max_features_sci.text())
        except Exception as e:
            QMessageBox.warning(self, "Input Error", f"Error parsing hyperparameter inputs: {e}")
            return None

        # Validate that all input lists are non-empty
        if not all([max_iter_values, learning_rate_values, max_depth_values, num_leaves_values, max_features_values]):
            QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
            return None

        # Check if all lengths of hyperparameter lists are equal
        lengths = {
            'max_iter': len(max_iter_values),
            'learning_rate': len(learning_rate_values),
            'max_depth': len(max_depth_values),
            'max_leaf_nodes': len(num_leaves_values),
            'max_features': len(max_features_values)
        }

        if len(set(lengths.values())) != 1:
            QMessageBox.warning(self, "Input Error", "Number of values for each hyperparameter must be equal.")
            return None

        try:
            # Create hyperparameters grid based on user input
            hyperparameters_grid = {
                'max_iter': [int(value) for value in max_iter_values],
                'learning_rate': [float(value) for value in learning_rate_values],
                'max_depth': [int(value) for value in max_depth_values],
                'max_leaf_nodes': [int(eval(value)) for value in num_leaves_values],
                'max_features': [float(value) for value in max_features_values]
            }
        except ValueError as ve:
            QMessageBox.warning(self, "Input Error", f"ValueError: {ve}")
            return None
        except (TypeError, NameError, SyntaxError) as e:
            QMessageBox.warning(self, "Input Error", f"Error evaluating values: {e}")
            return None
        except Exception as e:
            QMessageBox.warning(self, "Unexpected Error", f"Unexpected error: {e}")
            return None

        # Clear the input fields
        self.max_iter_sci.clear()
        self.learning_rate_sci.clear()
        self.max_depth_sci.clear()
        self.max_leaf_nodes_sci.clear()
        self.max_features_sci.clear()

        return hyperparameters_grid  
                                  
    def create_lgbm_hyperparameter_grid(self):
        """
        Creates a hyperparameter grid for LGBMRegressor based on user input.

        Returns:
            dict or None: Hyperparameter grid if successful, None otherwise.
        """
        # Retrieve hyperparameter values from GUI inputs
        try:
            n_estimators_values = re.split(r'\s|[,;]', self.n_estimators.text())
            learning_rate_values = re.split(r'\s|[,;]', self.learning_rate.text())
            max_depth_values = re.split(r'\s|[,;]', self.max_depth.text())
            num_leaves_values = re.split(r'\s|[,;]', self.num_leaves.text())
            colsample_bytree_values = re.split(r'\s|[,;]', self.colsample_bytree.text())
        except Exception as e:
            QMessageBox.warning(self, "Input Error", f"Error parsing hyperparameter inputs: {e}")
            return None

        # Validate that all input lists are non-empty
        if not all([n_estimators_values, learning_rate_values, max_depth_values, num_leaves_values, colsample_bytree_values]):
            QMessageBox.warning(self, "Input Required", "Please fill in all hyperparameters.")
            return None

        # Check if all lengths of hyperparameter lists are equal
        lengths = {
            'n_estimators': len(n_estimators_values),
            'learning_rate': len(learning_rate_values),
            'max_depth': len(max_depth_values),
            'num_leaves': len(num_leaves_values),
            'colsample_bytree': len(colsample_bytree_values)
        }

        if len(set(lengths.values())) != 1:
            QMessageBox.warning(self, "Input Error", "Number of values for each hyperparameter must be equal.")
            return None

        try:
            # Create hyperparameters grid based on user input
            hyperparameters_grid = {
                'n_estimators': [int(value) for value in n_estimators_values],
                'learning_rate': [float(value) for value in learning_rate_values],
                'max_depth': [int(value) for value in max_depth_values],
                'num_leaves': [int(eval(value)) for value in num_leaves_values],
                'colsample_bytree': [float(value) for value in colsample_bytree_values if value.strip()]
            }
        except ValueError as ve:
            QMessageBox.warning(self, "Input Error", f"ValueError: {ve}")
            return None
        except (TypeError, NameError, SyntaxError) as e:
            QMessageBox.warning(self, "Input Error", f"Error evaluating values: {e}")
            return None
        except Exception as e:
            QMessageBox.warning(self, "Unexpected Error", f"Unexpected error: {e}")
            return None

        # Clear the input fields
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
        self.suppress_errors = False  # Flag to control error suppression

    def write(self, message):
        # Check if we need to suppress errors based on the message content
        if self.suppress_errors and "Error" in message:
            return  # Suppress this message

        # Apply a custom style for error messages
        self.text_widget.append(f'<span style="color:red;">{message}</span>')
        self.text_widget.ensureCursorVisible()  # Ensure cursor is visible

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

