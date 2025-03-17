from qfluentwidgets import (
    PushButton,
    ComboBox,
    BodyLabel,
    LineEdit,
    TableWidget,
    ScrollArea,
)

from vnpy_evo.event import Event, EventEngine
from vnpy_evo.trader.engine import MainEngine
from vnpy_evo.trader.ui import QtCore, QtGui, QtWidgets
from vnpy_evo.trader.ui.monitor import (
    MsgCell,
    TimeCell,
    BaseMonitor
)
from ..base import (
    APP_NAME,
    EVENT_NOVA_LOG,
    EVENT_NOVA_STRATEGY
)
from ..engine import StrategyEngine


class NovaStrategyManager(QtWidgets.QWidget):
    """Strategy manager widget"""

    signal_strategy: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.strategy_engine: StrategyEngine = main_engine.get_engine(APP_NAME)

        self.widgets: dict[str, StraetgyWidget] = {}

        self.init_ui()
        self.register_event()
        self.strategy_engine.init_engine()
        self.update_class_combo()

    def init_ui(self) -> None:
        """Initialize UI"""
        self.setWindowTitle("Nova Strategy")

        # Create widgets
        self.class_combo: ComboBox = ComboBox()

        add_button: PushButton = PushButton("Add Strategy")
        add_button.clicked.connect(self.add_strategy)

        init_button: PushButton = PushButton("Initialize All")
        init_button.clicked.connect(self.strategy_engine.init_all_strategies)

        start_button: PushButton = PushButton("Start All")
        start_button.clicked.connect(self.strategy_engine.start_all_strategies)

        stop_button: PushButton = PushButton("Stop All")
        stop_button.clicked.connect(self.strategy_engine.stop_all_strategies)

        clear_button: PushButton = PushButton("Clear Logs")
        clear_button.clicked.connect(self.clear_log)

        self.scroll_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self.scroll_layout.addStretch()

        scroll_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        scroll_widget.setLayout(self.scroll_layout)

        scroll_area: ScrollArea = ScrollArea()
        scroll_area.setStyleSheet("background-color: transparent")
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        self.log_monitor: LogMonitor = LogMonitor(self.main_engine, self.event_engine)

        hbox1: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox1.addWidget(self.class_combo)
        hbox1.addWidget(add_button)
        hbox1.addStretch()
        hbox1.addWidget(init_button)
        hbox1.addWidget(start_button)
        hbox1.addWidget(stop_button)
        hbox1.addWidget(clear_button)

        hbox2: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox2.addWidget(scroll_area)
        hbox2.addWidget(self.log_monitor)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)

    def update_class_combo(self) -> None:
        """Update strategy class combobox"""
        self.class_combo.clear()
        self.class_combo.addItems(self.strategy_engine.get_all_strategy_class_names())

    def register_event(self) -> None:
        """Register event handler"""
        self.signal_strategy.connect(self.process_strategy_event)

        self.event_engine.register(EVENT_NOVA_STRATEGY, self.signal_strategy.emit)

    def process_strategy_event(self, event: Event) -> None:
        """Process strategy event update"""
        data: dict = event.data
        strategy_name: str = data["strategy_name"]

        if strategy_name in self.widgets:
            widget: StraetgyWidget = self.widgets[strategy_name]
            widget.update_data(data)
        else:
            widget: StraetgyWidget = StraetgyWidget(self, self.strategy_engine, data)
            self.scroll_layout.insertWidget(0, widget)
            self.widgets[strategy_name] = widget

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove strategy instance"""
        widget: StraetgyWidget = self.widgets.pop(strategy_name)
        widget.deleteLater()

    def add_strategy(self) -> None:
        """Add strategy instance"""
        class_name: str = str(self.class_combo.currentText())
        if not class_name:
            return

        parameters: dict = self.strategy_engine.get_strategy_class_parameters(class_name)
        editor: SettingEditor = SettingEditor(parameters, class_name=class_name)
        n: int = editor.exec_()

        if n == editor.DialogCode.Accepted:
            setting: dict = editor.get_setting()
            vt_symbols: str = setting.pop("vt_symbols").split(",")
            strategy_name: str = setting.pop("strategy_name")

            self.strategy_engine.add_strategy(
                class_name, strategy_name, vt_symbols, setting
            )

    def clear_log(self) -> None:
        """Clear log monitor"""
        self.log_monitor.setRowCount(0)

    def show(self) -> None:
        """Show maximized"""
        self.showMaximized()


class StraetgyWidget(QtWidgets.QFrame):
    """Widget of each strategy instance"""

    def __init__(
        self,
        strategy_manager: NovaStrategyManager,
        strategy_engine: StrategyEngine,
        data: dict
    ) -> None:
        """"""
        super().__init__()

        self.strategy_manager: NovaStrategyManager = strategy_manager
        self.strategy_engine: StrategyEngine = strategy_engine

        self.strategy_name: str = data["strategy_name"]
        self._data: dict = data

        self.init_ui()

    def init_ui(self) -> None:
        """初始化界面"""
        self.setFixedHeight(300)
        self.setFrameShape(self.Shape.Box)
        self.setLineWidth(3)

        self.status_label: BodyLabel = BodyLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.init_button: PushButton = PushButton("Initialize")
        self.init_button.clicked.connect(self.init_strategy)

        self.start_button: PushButton = PushButton("Start")
        self.start_button.clicked.connect(self.start_strategy)
        self.start_button.setEnabled(False)

        self.stop_button: PushButton = PushButton("Stop")
        self.stop_button.clicked.connect(self.stop_strategy)
        self.stop_button.setEnabled(False)

        self.edit_button: PushButton = PushButton("Edit")
        self.edit_button.clicked.connect(self.edit_strategy)

        self.remove_button: PushButton = PushButton("Remove")
        self.remove_button.clicked.connect(self.remove_strategy)

        strategy_name: str = self._data["strategy_name"]
        class_name: str = self._data["class_name"]
        author: str = self._data["author"]

        label_text: str = f"{strategy_name}  -  ({class_name} by {author})"
        label: BodyLabel = BodyLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)

        self.parameters_monitor: DataMonitor = DataMonitor(self._data["parameters"])
        self.variables_monitor: DataMonitor = DataMonitor(self._data["variables"])

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.status_label)
        hbox.addWidget(self.init_button)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        hbox.addWidget(self.edit_button)
        hbox.addWidget(self.remove_button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.parameters_monitor)
        vbox.addWidget(self.variables_monitor)
        self.setLayout(vbox)

    def update_data(self, data: dict) -> None:
        """Update strategy data"""
        self._data: dict = data

        self.parameters_monitor.update_data(data["parameters"])
        self.variables_monitor.update_data(data["variables"])

        # Update enable status of buttons
        inited: bool = data["inited"]
        trading: bool = data["trading"]

        if not inited:
            return
        self.init_button.setEnabled(False)
        self.status_label.setText("Inited")

        if trading:
            self.status_label.setText("Trading")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.edit_button.setEnabled(False)
            self.remove_button.setEnabled(False)
        else:
            self.status_label.setText("Stopped")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.edit_button.setEnabled(True)
            self.remove_button.setEnabled(True)

    def init_strategy(self) -> None:
        """Initialize strategy instance"""
        self.strategy_engine.init_strategy(self.strategy_name)

    def start_strategy(self) -> None:
        """Start strategy instance"""
        self.strategy_engine.start_strategy(self.strategy_name)

    def stop_strategy(self) -> None:
        """Stop strategy instance"""
        self.strategy_engine.stop_strategy(self.strategy_name)

    def edit_strategy(self) -> None:
        """Edit strategy parameters"""
        strategy_name: str = self._data["strategy_name"]

        parameters: dict = self.strategy_engine.get_strategy_parameters(strategy_name)
        editor: SettingEditor = SettingEditor(parameters, strategy_name=strategy_name)
        n: int = editor.exec_()

        if n == editor.DialogCode.Accepted:
            setting: dict = editor.get_setting()
            self.strategy_engine.edit_strategy(strategy_name, setting)

    def remove_strategy(self) -> None:
        """Remove strategy instance"""
        result: bool = self.strategy_engine.remove_strategy(self.strategy_name)

        # 只移除在策略引擎被成功移除的策略
        if result:
            self.strategy_manager.remove_strategy(self.strategy_name)


class DataMonitor(TableWidget):
    """Strategy data monitor for parameters and variables"""

    def __init__(self, data: dict) -> None:
        """"""
        super().__init__()

        self._data: dict = data
        self.cells: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """Initialize UI"""
        labels: list = list(self._data.keys())
        self.setColumnCount(len(labels))
        self.setHorizontalHeaderLabels(labels)

        self.setRowCount(1)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.EditTrigger.NoEditTriggers)

        for column, name in enumerate(self._data.keys()):
            value = self._data[name]

            cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(str(value))
            cell.setTextAlignment(QtCore.Qt.AlignCenter)

            self.setItem(0, column, cell)
            self.cells[name] = cell

    def update_data(self, data: dict):
        """Update data into table"""
        for name, value in data.items():
            cell: QtWidgets.QTableWidgetItem = self.cells[name]
            cell.setText(str(value))


class LogMonitor(BaseMonitor):
    """Strategy log monitor"""

    event_type: str = EVENT_NOVA_LOG
    data_key: str = ""
    sorting: bool = False

    headers: dict = {
        "time": {"display": "Time", "cell": TimeCell, "update": False},
        "msg": {"display": "Message", "cell": MsgCell, "update": False},
    }

    def init_ui(self) -> None:
        """Initialize UI"""
        super().init_ui()

        self.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

    def insert_new_row(self, data) -> None:
        """Insert a new row"""
        super().insert_new_row(data)
        self.resizeRowToContents(0)


class SettingEditor(QtWidgets.QDialog):
    """Strategy setting editor"""

    def __init__(
        self,
        parameters: dict,
        strategy_name: str = "",
        class_name: str = ""
    ) -> None:
        """"""
        super().__init__()

        self.parameters: dict = parameters
        self.strategy_name: str = strategy_name
        self.class_name: str = class_name

        self.edits: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """Initialize UI"""
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        if self.class_name:
            self.setWindowTitle(f"Add strategy: {self.class_name}")
            button_text: str = "Add"
            parameters: dict = {"strategy_name": "", "vt_symbols": ""}
            parameters.update(self.parameters)
        else:
            self.setWindowTitle(f"Edit parameters: {self.strategy_name}")
            button_text: str = "Confirm"
            parameters: dict = self.parameters

        for name, value in parameters.items():
            type_ = type(value)

            edit: LineEdit = LineEdit()
            edit.setText(str(value))

            if type_ is int:
                validator: QtGui.QIntValidator = QtGui.QIntValidator()
                edit.setValidator(validator)
            elif type_ is float:
                validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
                edit.setValidator(validator)

            form.addRow(f"{name} {type_}", edit)

            self.edits[name] = (edit, type_)

        button: PushButton = PushButton(button_text)
        button.clicked.connect(self.accept)
        form.addRow(button)

        self.setLayout(form)

    def get_setting(self) -> dict:
        """Get strategy setting dict"""
        setting: dict = {}

        if self.class_name:
            setting["class_name"] = self.class_name

        for name, tp in self.edits.items():
            edit, type_ = tp
            value_text = edit.text()

            if type_ == bool:
                if value_text == "True":
                    value: bool = True
                else:
                    value: bool = False
            else:
                value = type_(value_text)

            setting[name] = value

        return setting
