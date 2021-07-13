import sys
import copy
import numpy as np

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLayout, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from environment import GridEnv
from PolicyIteration import PolicyIteration
from ValueIteration import ValueIteration

class Demo(QWidget):
    """ a demo class, implemented with PyQt5 """

    def __init__(self, env):
        super(Demo, self).__init__()
        self.setWindowTitle('Demo: Policy Iteration & Value Iteration')

        """ environment introduction """
        self.env = env
        self.env_label = self.YaheiLabel('Grid World', 4)
        # environment figure
        self.env_matrix = np.zeros(shape=(env.nrows, env.ncols))
        self.env_matrix[0][0], self.env_matrix[env.nrows-1][env.ncols-1] = 1, 1    # destination
        self.env_figure = plt.figure()
        self.env_canvas = FigureCanvas(self.env_figure)
        self.env_ax = self.env_figure.add_subplot(111)
        self.env_ax.imshow(self.env_matrix, cmap=cm.binary)
        for r in range(env.nrows-1): self.env_ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(env.ncols-1): self.env_ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        self.env_canvas.draw()
        # environment description
        self.env_text = QLabel(self)
        self.env_text.setWordWrap(True)
        self.env_text.setText('Size\t:   %dx%d\n\n'%(env.nrows, env.ncols) + \
                              'Start\t:   random\n' + \
                              'End\t:   (0,0) or (%d,%d)\n\n'%(env.ncols-1, env.nrows-1) + \
                              'State\t:   position coordinates\n' + \
                              'Action\t:   {Up, Down, Left, Right}\n' + \
                              'Reward\t:   r=-1 for each step\n' + \
                              'Gamma\t:   γ=1\n')
        self.env_text.setFont(QFont("Microsoft Yahei", 10))
        # layout
        self.env_intro_layout = QHBoxLayout()
        self.env_intro_layout.addWidget(self.env_canvas)
        self.env_intro_layout.addSpacing(20)
        self.env_intro_layout.addWidget(self.env_text)

        """ Policy-Iteration demo """
        self.PI_label = self.YaheiLabel('Policy Iteration', 4)
        # buttons
        self.PI_button1 = QPushButton('Reset')
        self.PI_button1.clicked.connect(self.policy_iteration_reset)
        self.PI_button2 = QPushButton('Policy Evaluation')
        self.PI_button2.clicked.connect(self.policy_evaluation)
        self.PI_button3 = QPushButton('Policy Improvement')
        self.PI_button3.clicked.connect(self.policy_improvement)
        self.PI_button4 = QPushButton('Policy Iteration')
        self.PI_button4.clicked.connect(self.policy_iteration)
        self.PI_button5 = QPushButton('Save Figure')
        self.PI_button5.clicked.connect(lambda: self.figure_save(self.PI_values_figure))
        # layout of buttons
        self.PI_button_layout = QHBoxLayout()
        self.PI_button_layout.addWidget(self.PI_button1)
        self.PI_button_layout.addWidget(self.PI_button2)
        self.PI_button_layout.addWidget(self.PI_button3)
        self.PI_button_layout.addWidget(self.PI_button4)
        self.PI_button_layout.addWidget(self.PI_button5)
        # figure of state values
        self.PI = PolicyIteration(env)
        self.PI_values_figure = plt.figure()
        self.PI_values_canvas = FigureCanvas(self.PI_values_figure)
        self.PI_values_ax = self.PI_values_figure.add_subplot(121)
        self.PI_policy_ax = self.PI_values_figure.add_subplot(122)
        self.show_state_value(self.PI_values_ax, self.PI.values)
        self.show_policy(self.PI_policy_ax, self.PI.policy)
        self.PI_values_canvas.draw()
        # layout of values demo
        self.PI_values_layout = QVBoxLayout()
        self.PI_values_layout.addLayout(self.PI_button_layout)
        self.PI_values_layout.addWidget(self.PI_values_canvas)
        # demo of DP (decision process)
        self.PI_dp_env = copy.deepcopy(self.env)
        self.PI_dp_env.reset()
        self.PI_dp_figure = plt.figure()
        self.PI_dp_canvas = FigureCanvas(self.PI_dp_figure)
        self.PI_dp_ax = self.PI_dp_figure.add_subplot(111)
        self.PI_dp_ax.imshow(self.env_matrix, cmap=cm.binary)
        for r in range(self.PI_dp_env.nrows-1): self.PI_dp_ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(self.PI_dp_env.ncols-1): self.PI_dp_ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        self.PI_dp_ax.scatter([self.PI_dp_env.get_state()[0]], [self.PI_dp_env.get_state()[1]], marker='*', s=144, c='lightcoral')
        self.PI_dp_canvas.draw()
        self.PI_dp_timer = QTimer(self)
        self.PI_dp_timer.timeout.connect(lambda: self.decision_making_demo(
            self.PI_dp_env, self.PI.policy, self.PI_dp_ax, self.PI_dp_canvas))
        # button of DP (decision process) demo
        self.PI_dp_button_start = QPushButton('Start: Decision Process')
        self.PI_dp_button_start.clicked.connect(lambda: self.PI_dp_timer.start(1000))
        self.PI_dp_button_stop = QPushButton('Stop')
        self.PI_dp_button_stop.clicked.connect(self.PI_dp_timer.stop)
        # layout of DP buttons
        self.PI_dp_button_layout = QHBoxLayout()
        self.PI_dp_button_layout.addWidget(self.PI_dp_button_start)
        self.PI_dp_button_layout.addWidget(self.PI_dp_button_stop)
        # layout of DP demo
        self.PI_dp_layout = QVBoxLayout()
        self.PI_dp_layout.addLayout(self.PI_dp_button_layout)
        self.PI_dp_layout.addWidget(self.PI_dp_canvas)
        # layout of total Policy-Iteration demo
        self.PI_layout = QHBoxLayout()
        self.PI_layout.addLayout(self.PI_values_layout)
        self.PI_layout.addSpacing(20)
        self.PI_layout.addLayout(self.PI_dp_layout)
        self.PI_layout.setStretch(0, 20)
        self.PI_layout.setStretch(2, 9)

        """ Value-Iteration Demo """
        self.VI_label = self.YaheiLabel('Value Iteration', 4)
        # buttons
        self.VI_button1 = QPushButton('Reset')
        self.VI_button1.clicked.connect(self.value_iteration_reset)
        self.VI_button2 = QPushButton('Value Iteration (1 step)')
        self.VI_button2.clicked.connect(self.value_iteration_step)
        self.VI_button3 = QPushButton('Value Iteration')
        self.VI_button3.clicked.connect(self.value_iteration)
        self.VI_button4 = QPushButton('Save Figure')
        self.VI_button4.clicked.connect(lambda: self.figure_save(self.VI_values_figure))
        # layout of buttons
        self.VI_button_layout = QHBoxLayout()
        self.VI_button_layout.addWidget(self.VI_button1)
        self.VI_button_layout.addWidget(self.VI_button2)
        self.VI_button_layout.addWidget(self.VI_button3)
        self.VI_button_layout.addWidget(self.VI_button4)
        # figure of state values
        self.VI = ValueIteration(env)
        self.VI_values_figure = plt.figure()
        self.VI_values_canvas = FigureCanvas(self.VI_values_figure)
        self.VI_values_ax = self.VI_values_figure.add_subplot(121)
        self.VI_policy_ax = self.VI_values_figure.add_subplot(122)
        self.show_state_value(self.VI_values_ax, self.VI.values)
        self.show_policy(self.VI_policy_ax, self.VI.policy)
        self.VI_values_canvas.draw()
        # layout of values demo
        self.VI_values_layout = QVBoxLayout()
        self.VI_values_layout.addLayout(self.VI_button_layout)
        self.VI_values_layout.addWidget(self.VI_values_canvas)
        # demo of DP (decision process)
        self.VI_dp_env = copy.deepcopy(self.env)
        self.VI_dp_env.reset()
        self.VI_dp_figure = plt.figure()
        self.VI_dp_canvas = FigureCanvas(self.VI_dp_figure)
        self.VI_dp_ax = self.VI_dp_figure.add_subplot(111)
        self.VI_dp_ax.imshow(self.env_matrix, cmap=cm.binary)
        for r in range(self.VI_dp_env.nrows-1): self.VI_dp_ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(self.VI_dp_env.ncols-1): self.VI_dp_ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        self.VI_dp_ax.scatter([self.VI_dp_env.get_state()[0]], [self.VI_dp_env.get_state()[1]], marker='*', s=144, c='lightcoral')
        self.VI_dp_canvas.draw()
        self.VI_dp_timer = QTimer(self)
        self.VI_dp_timer.timeout.connect(lambda: self.decision_making_demo(
            self.VI_dp_env, self.VI.policy, self.VI_dp_ax, self.VI_dp_canvas))
        # button of DP (decision process) demo
        self.VI_dp_button_start = QPushButton('Start: Decision Process')
        self.VI_dp_button_start.clicked.connect(lambda: self.VI_dp_timer.start(1000))
        self.VI_dp_button_stop = QPushButton('Stop')
        self.VI_dp_button_stop.clicked.connect(self.VI_dp_timer.stop)
        # layout of DP buttons
        self.VI_dp_button_layout = QHBoxLayout()
        self.VI_dp_button_layout.addWidget(self.VI_dp_button_start)
        self.VI_dp_button_layout.addWidget(self.VI_dp_button_stop)
        # layout of DP demo
        self.VI_dp_layout = QVBoxLayout()
        self.VI_dp_layout.addLayout(self.VI_dp_button_layout)
        self.VI_dp_layout.addWidget(self.VI_dp_canvas)
        # layout of total Policy-Iteration demo
        self.VI_layout = QHBoxLayout()
        self.VI_layout.addLayout(self.VI_values_layout)
        self.VI_layout.addSpacing(20)
        self.VI_layout.addLayout(self.VI_dp_layout)
        self.VI_layout.setStretch(0, 20)
        self.VI_layout.setStretch(2, 9)

        self.all_v_layout = QVBoxLayout()
        self.all_v_layout.addWidget(self.env_label)
        self.all_v_layout.addLayout(self.env_intro_layout)
        self.all_v_layout.addSpacing(20)
        self.all_v_layout.addWidget(self.PI_label)
        self.all_v_layout.addLayout(self.PI_layout)
        self.all_v_layout.addSpacing(20)
        self.all_v_layout.addWidget(self.VI_label)
        self.all_v_layout.addLayout(self.VI_layout)
        self.total_widget = QWidget()
        self.total_widget.setLayout(self.all_v_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.total_widget)

        self.all_layout = QHBoxLayout()
        self.all_layout.addWidget(self.scroll)
        self.setLayout(self.all_layout)
        self.setMinimumSize(300, 1000)

    def YaheiLabel(self, text, size):
        """ QLabel with font Microsoft Yahei """
        return QLabel('<h%d style="font-family:Microsoft Yahei">%s</h%d>'%(size, text, size), self)

    def policy_evaluation(self):
        self.PI.policy_evaluation()
        self.PI_values_ax.cla()
        self.show_state_value(self.PI_values_ax, self.PI.values)
        self.PI_values_canvas.draw()

    def policy_improvement(self):
        self.PI.policy_improvement()
        self.PI_policy_ax.cla()
        self.show_policy(self.PI_policy_ax, self.PI.policy)
        self.PI_values_canvas.draw()

    def policy_iteration(self):
        self.PI.policy_iteration()
        self.PI_values_ax.cla()
        self.PI_policy_ax.cla()
        self.show_state_value(self.PI_values_ax, self.PI.values)
        self.show_policy(self.PI_policy_ax, self.PI.policy)
        self.PI_values_canvas.draw()

    def policy_iteration_reset(self):
        self.PI.reset()
        self.PI_values_ax.cla()
        self.PI_policy_ax.cla()
        self.show_state_value(self.PI_values_ax, self.PI.values)
        self.show_policy(self.PI_policy_ax, self.PI.policy)
        self.PI_values_canvas.draw()

    def value_iteration_step(self):
        self.VI.value_iteration_step()
        self.VI_values_ax.cla()
        self.VI_policy_ax.cla()
        self.show_state_value(self.VI_values_ax, self.VI.values)
        self.show_policy(self.VI_policy_ax, self.VI.policy)
        self.VI_values_canvas.draw()

    def value_iteration(self):
        self.VI.value_iteration()
        self.VI_values_ax.cla()
        self.VI_policy_ax.cla()
        self.show_state_value(self.VI_values_ax, self.VI.values)
        self.show_policy(self.VI_policy_ax, self.VI.policy)
        self.VI_values_canvas.draw()

    def value_iteration_reset(self):
        self.VI.reset()
        self.VI_values_ax.cla()
        self.VI_policy_ax.cla()
        self.show_state_value(self.VI_values_ax, self.VI.values)
        self.show_policy(self.VI_policy_ax, self.VI.policy)
        self.VI_values_canvas.draw()

    def show_state_value(self, ax, values):
        """ show value of each state """
        ax.imshow(values, cmap=cm.PuBu)
        for r in range(env.nrows-1): ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(env.ncols-1): ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        for r in range(env.nrows):
            for c in range(env.ncols): ax.text(c, r, "%.2f"%values[r][c],
                horizontalalignment='center', verticalalignment='center', fontsize=8)

    def show_policy(self, ax, policy):
        """ show policy in each state """
        ax.imshow(np.zeros(shape=(env.nrows, env.ncols)), cmap=cm.binary)
        for r in range(env.nrows):
            for c in range(env.ncols):
                if policy[r][c][0] > 0: ax.arrow(c, r, 0, -0.3, width=0.03, fc='k', ec='k')     # up
                if policy[r][c][1] > 0: ax.arrow(c, r, 0, 0.3, width=0.03, fc='k', ec='k')      # down
                if policy[r][c][2] > 0: ax.arrow(c, r, -0.3, 0, width=0.03, fc='k', ec='k')     # left
                if policy[r][c][3] > 0: ax.arrow(c, r, 0.3, 0, width=0.03, fc='k', ec='k')      # right
        for r in range(env.nrows-1): ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(env.ncols-1): ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        ax.axis([-0.5, 7.5, -0.5, 7.5])
        ax.invert_yaxis()

    def figure_save(self, figure):
        filename, _ = QFileDialog.getSaveFileName(None, "Save Figure", "./")
        figure.savefig(filename, dpi=500)

    def decision_making_demo(self, env, policy, ax, canvas):
        """ one step of decision making """
        obs = env.get_state()
        action = np.random.choice(np.arange(env.nactions), p=policy[obs[1]][obs[0]])
        next_obs, reward, done = env.step(action)
        if done: env.reset()

        # demo
        ax.cla()
        ax.imshow(self.env_matrix, cmap=cm.binary)
        for r in range(env.nrows-1): ax.plot([-0.5, 7.5], [r+0.5, r+0.5], c='gray')
        for c in range(env.ncols-1): ax.plot([c+0.5, c+0.5], [-0.5, 7.5], c='gray')
        ax.scatter([env.get_state()[0]], [env.get_state()[1]], marker='*', s=144, c='lightcoral')
        canvas.draw()

def main(env):
    app = QApplication(sys.argv)
    demo = Demo(env)
    demo.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    env = GridEnv(nrows=8, ncols=8)
    main(env)