#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on Wed Nov 18 22:53:37 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.4'
expName = 'assignment3'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'age': '',
    'Gender': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1440,900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = 'data/' + expInfo['participant'] + '_' + expName + '_' + expInfo['date']
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/shubhamyadav/Desktop/assignment3.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color='black', colorSpace='named',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'named'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "RSVP_exp_start" ---
    Instructions = visual.TextStim(win=win, name='Instructions',
        text='You will be displayed 15 words,  \n\nOne of those word will be coloured RED\nYou have to concentrate on the next word shown immediately after that.\n\nOne MCQ question will be shown in which you have to select the word shown after coloured word.\n\npress Space to continue\n\n',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    next_start = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "Plus" ---
    plus = visual.TextStim(win=win, name='plus',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "RSVP" ---
    text1 = visual.TextStim(win=win, name='text1',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text2 = visual.TextStim(win=win, name='text2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text3 = visual.TextStim(win=win, name='text3',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text4 = visual.TextStim(win=win, name='text4',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text5 = visual.TextStim(win=win, name='text5',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text6 = visual.TextStim(win=win, name='text6',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text7 = visual.TextStim(win=win, name='text7',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    text8 = visual.TextStim(win=win, name='text8',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    text9 = visual.TextStim(win=win, name='text9',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text10 = visual.TextStim(win=win, name='text10',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    text11 = visual.TextStim(win=win, name='text11',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    text12 = visual.TextStim(win=win, name='text12',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    text13 = visual.TextStim(win=win, name='text13',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    text14 = visual.TextStim(win=win, name='text14',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    text15 = visual.TextStim(win=win, name='text15',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "mcq" ---
    text = visual.TextStim(win=win, name='text',
        text='Which word did you see?',
        font='Arial',
        pos=[0,0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mcq1 = visual.TextStim(win=win, name='mcq1',
        text='',
        font='Arial',
        pos=[0,0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mcq2 = visual.TextStim(win=win, name='mcq2',
        text='',
        font='Arial',
        pos=[0,0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mcq3 = visual.TextStim(win=win, name='mcq3',
        text='',
        font='Arial',
        pos=[0,-0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mcq4 = visual.TextStim(win=win, name='mcq4',
        text='',
        font='Arial',
        pos=[0,-0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "break_2" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Thats All for Trail\n\nPlease take a break.\n\nPress Space to continue to start Experiment',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "Plus" ---
    plus = visual.TextStim(win=win, name='plus',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "RSVP" ---
    text1 = visual.TextStim(win=win, name='text1',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text2 = visual.TextStim(win=win, name='text2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text3 = visual.TextStim(win=win, name='text3',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text4 = visual.TextStim(win=win, name='text4',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text5 = visual.TextStim(win=win, name='text5',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text6 = visual.TextStim(win=win, name='text6',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text7 = visual.TextStim(win=win, name='text7',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    text8 = visual.TextStim(win=win, name='text8',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    text9 = visual.TextStim(win=win, name='text9',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text10 = visual.TextStim(win=win, name='text10',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    text11 = visual.TextStim(win=win, name='text11',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    text12 = visual.TextStim(win=win, name='text12',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    text13 = visual.TextStim(win=win, name='text13',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    text14 = visual.TextStim(win=win, name='text14',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    text15 = visual.TextStim(win=win, name='text15',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "mcq" ---
    text = visual.TextStim(win=win, name='text',
        text='Which word did you see?',
        font='Arial',
        pos=[0,0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mcq1 = visual.TextStim(win=win, name='mcq1',
        text='',
        font='Arial',
        pos=[0,0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mcq2 = visual.TextStim(win=win, name='mcq2',
        text='',
        font='Arial',
        pos=[0,0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mcq3 = visual.TextStim(win=win, name='mcq3',
        text='',
        font='Arial',
        pos=[0,-0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mcq4 = visual.TextStim(win=win, name='mcq4',
        text='',
        font='Arial',
        pos=[0,-0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "break_3" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Thanks for Participating\n\nPlease take a break.\n\nPress Space to continue to continue Experiment',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "Plus" ---
    plus = visual.TextStim(win=win, name='plus',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "RSVP" ---
    text1 = visual.TextStim(win=win, name='text1',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text2 = visual.TextStim(win=win, name='text2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text3 = visual.TextStim(win=win, name='text3',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text4 = visual.TextStim(win=win, name='text4',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text5 = visual.TextStim(win=win, name='text5',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text6 = visual.TextStim(win=win, name='text6',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text7 = visual.TextStim(win=win, name='text7',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    text8 = visual.TextStim(win=win, name='text8',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    text9 = visual.TextStim(win=win, name='text9',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text10 = visual.TextStim(win=win, name='text10',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    text11 = visual.TextStim(win=win, name='text11',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    text12 = visual.TextStim(win=win, name='text12',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    text13 = visual.TextStim(win=win, name='text13',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    text14 = visual.TextStim(win=win, name='text14',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    text15 = visual.TextStim(win=win, name='text15',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "mcq" ---
    text = visual.TextStim(win=win, name='text',
        text='Which word did you see?',
        font='Arial',
        pos=[0,0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mcq1 = visual.TextStim(win=win, name='mcq1',
        text='',
        font='Arial',
        pos=[0,0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mcq2 = visual.TextStim(win=win, name='mcq2',
        text='',
        font='Arial',
        pos=[0,0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mcq3 = visual.TextStim(win=win, name='mcq3',
        text='',
        font='Arial',
        pos=[0,-0.05], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mcq4 = visual.TextStim(win=win, name='mcq4',
        text='',
        font='Arial',
        pos=[0,-0.15], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "finish" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Its done bro\n\nThanks For your Time',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "RSVP_exp_start" ---
    # create an object to store info about Routine RSVP_exp_start
    RSVP_exp_start = data.Routine(
        name='RSVP_exp_start',
        components=[Instructions, next_start],
    )
    RSVP_exp_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for next_start
    next_start.keys = []
    next_start.rt = []
    _next_start_allKeys = []
    # store start times for RSVP_exp_start
    RSVP_exp_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    RSVP_exp_start.tStart = globalClock.getTime(format='float')
    RSVP_exp_start.status = STARTED
    thisExp.addData('RSVP_exp_start.started', RSVP_exp_start.tStart)
    RSVP_exp_start.maxDuration = None
    # keep track of which components have finished
    RSVP_exp_startComponents = RSVP_exp_start.components
    for thisComponent in RSVP_exp_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "RSVP_exp_start" ---
    thisExp.currentRoutine = RSVP_exp_start
    RSVP_exp_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions* updates
        
        # if Instructions is starting this frame...
        if Instructions.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Instructions.frameNStart = frameN  # exact frame index
            Instructions.tStart = t  # local t and not account for scr refresh
            Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instructions.started')
            # update status
            Instructions.status = STARTED
            Instructions.setAutoDraw(True)
        
        # if Instructions is active this frame...
        if Instructions.status == STARTED:
            # update params
            pass
        
        # *next_start* updates
        waitOnFlip = False
        
        # if next_start is starting this frame...
        if next_start.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            next_start.frameNStart = frameN  # exact frame index
            next_start.tStart = t  # local t and not account for scr refresh
            next_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'next_start.started')
            # update status
            next_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(next_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(next_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if next_start.status == STARTED and not waitOnFlip:
            theseKeys = next_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _next_start_allKeys.extend(theseKeys)
            if len(_next_start_allKeys):
                next_start.keys = _next_start_allKeys[-1].name  # just the last key pressed
                next_start.rt = _next_start_allKeys[-1].rt
                next_start.duration = _next_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=RSVP_exp_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            RSVP_exp_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if RSVP_exp_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in RSVP_exp_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "RSVP_exp_start" ---
    for thisComponent in RSVP_exp_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for RSVP_exp_start
    RSVP_exp_start.tStop = globalClock.getTime(format='float')
    RSVP_exp_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('RSVP_exp_start.stopped', RSVP_exp_start.tStop)
    # check responses
    if next_start.keys in ['', [], None]:  # No response was made
        next_start.keys = None
    thisExp.addData('next_start.keys',next_start.keys)
    if next_start.keys != None:  # we had a response
        thisExp.addData('next_start.rt', next_start.rt)
        thisExp.addData('next_start.duration', next_start.duration)
    thisExp.nextEntry()
    # the Routine "RSVP_exp_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_run = data.TrialHandler2(
        name='trial_run',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trail_run.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trial_run)  # add the loop to the experiment
    thisTrial_run = trial_run.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_run.rgb)
    if thisTrial_run != None:
        for paramName in thisTrial_run:
            globals()[paramName] = thisTrial_run[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_run in trial_run:
        trial_run.status = STARTED
        if hasattr(thisTrial_run, 'status'):
            thisTrial_run.status = STARTED
        currentLoop = trial_run
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_run.rgb)
        if thisTrial_run != None:
            for paramName in thisTrial_run:
                globals()[paramName] = thisTrial_run[paramName]
        
        # --- Prepare to start Routine "Plus" ---
        # create an object to store info about Routine Plus
        Plus = data.Routine(
            name='Plus',
            components=[plus],
        )
        Plus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Plus
        Plus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Plus.tStart = globalClock.getTime(format='float')
        Plus.status = STARTED
        thisExp.addData('Plus.started', Plus.tStart)
        Plus.maxDuration = None
        # keep track of which components have finished
        PlusComponents = Plus.components
        for thisComponent in Plus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Plus" ---
        thisExp.currentRoutine = Plus
        Plus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_run, 'status') and thisTrial_run.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *plus* updates
            
            # if plus is starting this frame...
            if plus.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                plus.frameNStart = frameN  # exact frame index
                plus.tStart = t  # local t and not account for scr refresh
                plus.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(plus, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'plus.started')
                # update status
                plus.status = STARTED
                plus.setAutoDraw(True)
            
            # if plus is active this frame...
            if plus.status == STARTED:
                # update params
                pass
            
            # if plus is stopping this frame...
            if plus.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > plus.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    plus.tStop = t  # not accounting for scr refresh
                    plus.tStopRefresh = tThisFlipGlobal  # on global time
                    plus.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'plus.stopped')
                    # update status
                    plus.status = FINISHED
                    plus.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Plus,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Plus.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Plus.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Plus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Plus" ---
        for thisComponent in Plus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Plus
        Plus.tStop = globalClock.getTime(format='float')
        Plus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Plus.stopped', Plus.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Plus.maxDurationReached:
            routineTimer.addTime(-Plus.maxDuration)
        elif Plus.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "RSVP" ---
        # create an object to store info about Routine RSVP
        RSVP = data.Routine(
            name='RSVP',
            components=[text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15],
        )
        RSVP.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text1.setColor(c_1, colorSpace='rgb')
        text1.setText(w_1
        )
        text2.setColor(c_2, colorSpace='rgb')
        text2.setText(w_2)
        text3.setColor(c_3, colorSpace='rgb')
        text3.setText(w_3)
        text4.setColor(c_4, colorSpace='rgb')
        text4.setText(w_4)
        text5.setColor(c_5, colorSpace='rgb')
        text5.setText(w_5)
        text6.setColor(c_6, colorSpace='rgb')
        text6.setText(w_6)
        text7.setColor(c_7, colorSpace='rgb')
        text7.setText(w_7)
        text8.setColor(c_8, colorSpace='rgb')
        text8.setText(w_8)
        text9.setColor(c_9, colorSpace='rgb')
        text9.setText(w_9)
        text10.setColor(c_10, colorSpace='rgb')
        text10.setText(w_10)
        text11.setColor(c_11, colorSpace='rgb')
        text11.setText(w_11)
        text12.setColor(c_12, colorSpace='rgb')
        text12.setText(w_12)
        text13.setColor(c_13, colorSpace='rgb')
        text13.setText(w_13)
        text14.setColor(c_14, colorSpace='rgb')
        text14.setText(w_14)
        text15.setColor(c_15, colorSpace='rgb')
        text15.setText(w_15)
        # store start times for RSVP
        RSVP.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        RSVP.tStart = globalClock.getTime(format='float')
        RSVP.status = STARTED
        thisExp.addData('RSVP.started', RSVP.tStart)
        RSVP.maxDuration = None
        # keep track of which components have finished
        RSVPComponents = RSVP.components
        for thisComponent in RSVP.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RSVP" ---
        thisExp.currentRoutine = RSVP
        RSVP.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.25:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_run, 'status') and thisTrial_run.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text1* updates
            
            # if text1 is starting this frame...
            if text1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text1.frameNStart = frameN  # exact frame index
                text1.tStart = t  # local t and not account for scr refresh
                text1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text1.started')
                # update status
                text1.status = STARTED
                text1.setAutoDraw(True)
            
            # if text1 is active this frame...
            if text1.status == STARTED:
                # update params
                pass
            
            # if text1 is stopping this frame...
            if text1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text1.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text1.tStop = t  # not accounting for scr refresh
                    text1.tStopRefresh = tThisFlipGlobal  # on global time
                    text1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text1.stopped')
                    # update status
                    text1.status = FINISHED
                    text1.setAutoDraw(False)
            
            # *text2* updates
            
            # if text2 is starting this frame...
            if text2.status == NOT_STARTED and tThisFlip >= 0.15-frameTolerance:
                # keep track of start time/frame for later
                text2.frameNStart = frameN  # exact frame index
                text2.tStart = t  # local t and not account for scr refresh
                text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text2.started')
                # update status
                text2.status = STARTED
                text2.setAutoDraw(True)
            
            # if text2 is active this frame...
            if text2.status == STARTED:
                # update params
                pass
            
            # if text2 is stopping this frame...
            if text2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text2.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text2.tStop = t  # not accounting for scr refresh
                    text2.tStopRefresh = tThisFlipGlobal  # on global time
                    text2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text2.stopped')
                    # update status
                    text2.status = FINISHED
                    text2.setAutoDraw(False)
            
            # *text3* updates
            
            # if text3 is starting this frame...
            if text3.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                text3.frameNStart = frameN  # exact frame index
                text3.tStart = t  # local t and not account for scr refresh
                text3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text3.started')
                # update status
                text3.status = STARTED
                text3.setAutoDraw(True)
            
            # if text3 is active this frame...
            if text3.status == STARTED:
                # update params
                pass
            
            # if text3 is stopping this frame...
            if text3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text3.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text3.tStop = t  # not accounting for scr refresh
                    text3.tStopRefresh = tThisFlipGlobal  # on global time
                    text3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text3.stopped')
                    # update status
                    text3.status = FINISHED
                    text3.setAutoDraw(False)
            
            # *text4* updates
            
            # if text4 is starting this frame...
            if text4.status == NOT_STARTED and tThisFlip >= 0.45-frameTolerance:
                # keep track of start time/frame for later
                text4.frameNStart = frameN  # exact frame index
                text4.tStart = t  # local t and not account for scr refresh
                text4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text4.started')
                # update status
                text4.status = STARTED
                text4.setAutoDraw(True)
            
            # if text4 is active this frame...
            if text4.status == STARTED:
                # update params
                pass
            
            # if text4 is stopping this frame...
            if text4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text4.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text4.tStop = t  # not accounting for scr refresh
                    text4.tStopRefresh = tThisFlipGlobal  # on global time
                    text4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text4.stopped')
                    # update status
                    text4.status = FINISHED
                    text4.setAutoDraw(False)
            
            # *text5* updates
            
            # if text5 is starting this frame...
            if text5.status == NOT_STARTED and tThisFlip >= 0.6-frameTolerance:
                # keep track of start time/frame for later
                text5.frameNStart = frameN  # exact frame index
                text5.tStart = t  # local t and not account for scr refresh
                text5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text5.started')
                # update status
                text5.status = STARTED
                text5.setAutoDraw(True)
            
            # if text5 is active this frame...
            if text5.status == STARTED:
                # update params
                pass
            
            # if text5 is stopping this frame...
            if text5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text5.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text5.tStop = t  # not accounting for scr refresh
                    text5.tStopRefresh = tThisFlipGlobal  # on global time
                    text5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text5.stopped')
                    # update status
                    text5.status = FINISHED
                    text5.setAutoDraw(False)
            
            # *text6* updates
            
            # if text6 is starting this frame...
            if text6.status == NOT_STARTED and tThisFlip >= 0.75-frameTolerance:
                # keep track of start time/frame for later
                text6.frameNStart = frameN  # exact frame index
                text6.tStart = t  # local t and not account for scr refresh
                text6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text6.started')
                # update status
                text6.status = STARTED
                text6.setAutoDraw(True)
            
            # if text6 is active this frame...
            if text6.status == STARTED:
                # update params
                pass
            
            # if text6 is stopping this frame...
            if text6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text6.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text6.tStop = t  # not accounting for scr refresh
                    text6.tStopRefresh = tThisFlipGlobal  # on global time
                    text6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text6.stopped')
                    # update status
                    text6.status = FINISHED
                    text6.setAutoDraw(False)
            
            # *text7* updates
            
            # if text7 is starting this frame...
            if text7.status == NOT_STARTED and tThisFlip >= 0.9-frameTolerance:
                # keep track of start time/frame for later
                text7.frameNStart = frameN  # exact frame index
                text7.tStart = t  # local t and not account for scr refresh
                text7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text7.started')
                # update status
                text7.status = STARTED
                text7.setAutoDraw(True)
            
            # if text7 is active this frame...
            if text7.status == STARTED:
                # update params
                pass
            
            # if text7 is stopping this frame...
            if text7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text7.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text7.tStop = t  # not accounting for scr refresh
                    text7.tStopRefresh = tThisFlipGlobal  # on global time
                    text7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text7.stopped')
                    # update status
                    text7.status = FINISHED
                    text7.setAutoDraw(False)
            
            # *text8* updates
            
            # if text8 is starting this frame...
            if text8.status == NOT_STARTED and tThisFlip >= 1.05-frameTolerance:
                # keep track of start time/frame for later
                text8.frameNStart = frameN  # exact frame index
                text8.tStart = t  # local t and not account for scr refresh
                text8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text8.started')
                # update status
                text8.status = STARTED
                text8.setAutoDraw(True)
            
            # if text8 is active this frame...
            if text8.status == STARTED:
                # update params
                pass
            
            # if text8 is stopping this frame...
            if text8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text8.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text8.tStop = t  # not accounting for scr refresh
                    text8.tStopRefresh = tThisFlipGlobal  # on global time
                    text8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text8.stopped')
                    # update status
                    text8.status = FINISHED
                    text8.setAutoDraw(False)
            
            # *text9* updates
            
            # if text9 is starting this frame...
            if text9.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                text9.frameNStart = frameN  # exact frame index
                text9.tStart = t  # local t and not account for scr refresh
                text9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text9.started')
                # update status
                text9.status = STARTED
                text9.setAutoDraw(True)
            
            # if text9 is active this frame...
            if text9.status == STARTED:
                # update params
                pass
            
            # if text9 is stopping this frame...
            if text9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text9.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text9.tStop = t  # not accounting for scr refresh
                    text9.tStopRefresh = tThisFlipGlobal  # on global time
                    text9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text9.stopped')
                    # update status
                    text9.status = FINISHED
                    text9.setAutoDraw(False)
            
            # *text10* updates
            
            # if text10 is starting this frame...
            if text10.status == NOT_STARTED and tThisFlip >= 1.35-frameTolerance:
                # keep track of start time/frame for later
                text10.frameNStart = frameN  # exact frame index
                text10.tStart = t  # local t and not account for scr refresh
                text10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text10.started')
                # update status
                text10.status = STARTED
                text10.setAutoDraw(True)
            
            # if text10 is active this frame...
            if text10.status == STARTED:
                # update params
                pass
            
            # if text10 is stopping this frame...
            if text10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text10.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text10.tStop = t  # not accounting for scr refresh
                    text10.tStopRefresh = tThisFlipGlobal  # on global time
                    text10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text10.stopped')
                    # update status
                    text10.status = FINISHED
                    text10.setAutoDraw(False)
            
            # *text11* updates
            
            # if text11 is starting this frame...
            if text11.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                text11.frameNStart = frameN  # exact frame index
                text11.tStart = t  # local t and not account for scr refresh
                text11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text11.started')
                # update status
                text11.status = STARTED
                text11.setAutoDraw(True)
            
            # if text11 is active this frame...
            if text11.status == STARTED:
                # update params
                pass
            
            # if text11 is stopping this frame...
            if text11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text11.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text11.tStop = t  # not accounting for scr refresh
                    text11.tStopRefresh = tThisFlipGlobal  # on global time
                    text11.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text11.stopped')
                    # update status
                    text11.status = FINISHED
                    text11.setAutoDraw(False)
            
            # *text12* updates
            
            # if text12 is starting this frame...
            if text12.status == NOT_STARTED and tThisFlip >= 1.65-frameTolerance:
                # keep track of start time/frame for later
                text12.frameNStart = frameN  # exact frame index
                text12.tStart = t  # local t and not account for scr refresh
                text12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text12.started')
                # update status
                text12.status = STARTED
                text12.setAutoDraw(True)
            
            # if text12 is active this frame...
            if text12.status == STARTED:
                # update params
                pass
            
            # if text12 is stopping this frame...
            if text12.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text12.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text12.tStop = t  # not accounting for scr refresh
                    text12.tStopRefresh = tThisFlipGlobal  # on global time
                    text12.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text12.stopped')
                    # update status
                    text12.status = FINISHED
                    text12.setAutoDraw(False)
            
            # *text13* updates
            
            # if text13 is starting this frame...
            if text13.status == NOT_STARTED and tThisFlip >= 1.8-frameTolerance:
                # keep track of start time/frame for later
                text13.frameNStart = frameN  # exact frame index
                text13.tStart = t  # local t and not account for scr refresh
                text13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text13.started')
                # update status
                text13.status = STARTED
                text13.setAutoDraw(True)
            
            # if text13 is active this frame...
            if text13.status == STARTED:
                # update params
                pass
            
            # if text13 is stopping this frame...
            if text13.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text13.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text13.tStop = t  # not accounting for scr refresh
                    text13.tStopRefresh = tThisFlipGlobal  # on global time
                    text13.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text13.stopped')
                    # update status
                    text13.status = FINISHED
                    text13.setAutoDraw(False)
            
            # *text14* updates
            
            # if text14 is starting this frame...
            if text14.status == NOT_STARTED and tThisFlip >= 1.95-frameTolerance:
                # keep track of start time/frame for later
                text14.frameNStart = frameN  # exact frame index
                text14.tStart = t  # local t and not account for scr refresh
                text14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text14.started')
                # update status
                text14.status = STARTED
                text14.setAutoDraw(True)
            
            # if text14 is active this frame...
            if text14.status == STARTED:
                # update params
                pass
            
            # if text14 is stopping this frame...
            if text14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text14.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text14.tStop = t  # not accounting for scr refresh
                    text14.tStopRefresh = tThisFlipGlobal  # on global time
                    text14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text14.stopped')
                    # update status
                    text14.status = FINISHED
                    text14.setAutoDraw(False)
            
            # *text15* updates
            
            # if text15 is starting this frame...
            if text15.status == NOT_STARTED and tThisFlip >= 2.1-frameTolerance:
                # keep track of start time/frame for later
                text15.frameNStart = frameN  # exact frame index
                text15.tStart = t  # local t and not account for scr refresh
                text15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text15.started')
                # update status
                text15.status = STARTED
                text15.setAutoDraw(True)
            
            # if text15 is active this frame...
            if text15.status == STARTED:
                # update params
                pass
            
            # if text15 is stopping this frame...
            if text15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text15.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text15.tStop = t  # not accounting for scr refresh
                    text15.tStopRefresh = tThisFlipGlobal  # on global time
                    text15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text15.stopped')
                    # update status
                    text15.status = FINISHED
                    text15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=RSVP,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                RSVP.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if RSVP.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in RSVP.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RSVP" ---
        for thisComponent in RSVP.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for RSVP
        RSVP.tStop = globalClock.getTime(format='float')
        RSVP.tStopRefresh = tThisFlipGlobal
        thisExp.addData('RSVP.stopped', RSVP.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if RSVP.maxDurationReached:
            routineTimer.addTime(-RSVP.maxDuration)
        elif RSVP.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.250000)
        
        # --- Prepare to start Routine "mcq" ---
        # create an object to store info about Routine mcq
        mcq = data.Routine(
            name='mcq',
            components=[text, mcq1, mcq2, mcq3, mcq4, key_resp],
        )
        mcq.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        mcq1.setText("1. " + str(mcq_option_1))
        mcq2.setText("2. " + str(mcq_option_2))
        mcq3.setText("3. " + str(mcq_option_3))
        mcq4.setText("4. " + str(mcq_option_4))
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for mcq
        mcq.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mcq.tStart = globalClock.getTime(format='float')
        mcq.status = STARTED
        thisExp.addData('mcq.started', mcq.tStart)
        mcq.maxDuration = None
        # keep track of which components have finished
        mcqComponents = mcq.components
        for thisComponent in mcq.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mcq" ---
        thisExp.currentRoutine = mcq
        mcq.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_run, 'status') and thisTrial_run.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *mcq1* updates
            
            # if mcq1 is starting this frame...
            if mcq1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq1.frameNStart = frameN  # exact frame index
                mcq1.tStart = t  # local t and not account for scr refresh
                mcq1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq1.started')
                # update status
                mcq1.status = STARTED
                mcq1.setAutoDraw(True)
            
            # if mcq1 is active this frame...
            if mcq1.status == STARTED:
                # update params
                pass
            
            # *mcq2* updates
            
            # if mcq2 is starting this frame...
            if mcq2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq2.frameNStart = frameN  # exact frame index
                mcq2.tStart = t  # local t and not account for scr refresh
                mcq2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq2.started')
                # update status
                mcq2.status = STARTED
                mcq2.setAutoDraw(True)
            
            # if mcq2 is active this frame...
            if mcq2.status == STARTED:
                # update params
                pass
            
            # *mcq3* updates
            
            # if mcq3 is starting this frame...
            if mcq3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq3.frameNStart = frameN  # exact frame index
                mcq3.tStart = t  # local t and not account for scr refresh
                mcq3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq3.started')
                # update status
                mcq3.status = STARTED
                mcq3.setAutoDraw(True)
            
            # if mcq3 is active this frame...
            if mcq3.status == STARTED:
                # update params
                pass
            
            # *mcq4* updates
            
            # if mcq4 is starting this frame...
            if mcq4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq4.frameNStart = frameN  # exact frame index
                mcq4.tStart = t  # local t and not account for scr refresh
                mcq4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq4.started')
                # update status
                mcq4.status = STARTED
                mcq4.setAutoDraw(True)
            
            # if mcq4 is active this frame...
            if mcq4.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct_key)) or (key_resp.keys == correct_key):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mcq,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mcq.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mcq.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mcq.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mcq" ---
        for thisComponent in mcq.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mcq
        mcq.tStop = globalClock.getTime(format='float')
        mcq.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mcq.stopped', mcq.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_run (TrialHandler)
        trial_run.addData('key_resp.keys',key_resp.keys)
        trial_run.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trial_run.addData('key_resp.rt', key_resp.rt)
            trial_run.addData('key_resp.duration', key_resp.duration)
        # the Routine "mcq" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial_run as finished
        if hasattr(thisTrial_run, 'status'):
            thisTrial_run.status = FINISHED
        # if awaiting a pause, pause now
        if trial_run.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trial_run.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trial_run'
    trial_run.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_2" ---
    # create an object to store info about Routine break_2
    break_2 = data.Routine(
        name='break_2',
        components=[text_2, key_resp_2],
    )
    break_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for break_2
    break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_2.tStart = globalClock.getTime(format='float')
    break_2.status = STARTED
    thisExp.addData('break_2.started', break_2.tStart)
    break_2.maxDuration = None
    # keep track of which components have finished
    break_2Components = break_2.components
    for thisComponent in break_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_2" ---
    thisExp.currentRoutine = break_2
    break_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=break_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            break_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if break_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_2" ---
    for thisComponent in break_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_2
    break_2.tStop = globalClock.getTime(format='float')
    break_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_2.stopped', break_2.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    run1 = data.TrialHandler2(
        name='run1',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('run11.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(run1)  # add the loop to the experiment
    thisRun1 = run1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun1.rgb)
    if thisRun1 != None:
        for paramName in thisRun1:
            globals()[paramName] = thisRun1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRun1 in run1:
        run1.status = STARTED
        if hasattr(thisRun1, 'status'):
            thisRun1.status = STARTED
        currentLoop = run1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRun1.rgb)
        if thisRun1 != None:
            for paramName in thisRun1:
                globals()[paramName] = thisRun1[paramName]
        
        # --- Prepare to start Routine "Plus" ---
        # create an object to store info about Routine Plus
        Plus = data.Routine(
            name='Plus',
            components=[plus],
        )
        Plus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Plus
        Plus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Plus.tStart = globalClock.getTime(format='float')
        Plus.status = STARTED
        thisExp.addData('Plus.started', Plus.tStart)
        Plus.maxDuration = None
        # keep track of which components have finished
        PlusComponents = Plus.components
        for thisComponent in Plus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Plus" ---
        thisExp.currentRoutine = Plus
        Plus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisRun1, 'status') and thisRun1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *plus* updates
            
            # if plus is starting this frame...
            if plus.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                plus.frameNStart = frameN  # exact frame index
                plus.tStart = t  # local t and not account for scr refresh
                plus.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(plus, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'plus.started')
                # update status
                plus.status = STARTED
                plus.setAutoDraw(True)
            
            # if plus is active this frame...
            if plus.status == STARTED:
                # update params
                pass
            
            # if plus is stopping this frame...
            if plus.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > plus.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    plus.tStop = t  # not accounting for scr refresh
                    plus.tStopRefresh = tThisFlipGlobal  # on global time
                    plus.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'plus.stopped')
                    # update status
                    plus.status = FINISHED
                    plus.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Plus,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Plus.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Plus.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Plus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Plus" ---
        for thisComponent in Plus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Plus
        Plus.tStop = globalClock.getTime(format='float')
        Plus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Plus.stopped', Plus.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Plus.maxDurationReached:
            routineTimer.addTime(-Plus.maxDuration)
        elif Plus.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "RSVP" ---
        # create an object to store info about Routine RSVP
        RSVP = data.Routine(
            name='RSVP',
            components=[text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15],
        )
        RSVP.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text1.setColor(c_1, colorSpace='rgb')
        text1.setText(w_1
        )
        text2.setColor(c_2, colorSpace='rgb')
        text2.setText(w_2)
        text3.setColor(c_3, colorSpace='rgb')
        text3.setText(w_3)
        text4.setColor(c_4, colorSpace='rgb')
        text4.setText(w_4)
        text5.setColor(c_5, colorSpace='rgb')
        text5.setText(w_5)
        text6.setColor(c_6, colorSpace='rgb')
        text6.setText(w_6)
        text7.setColor(c_7, colorSpace='rgb')
        text7.setText(w_7)
        text8.setColor(c_8, colorSpace='rgb')
        text8.setText(w_8)
        text9.setColor(c_9, colorSpace='rgb')
        text9.setText(w_9)
        text10.setColor(c_10, colorSpace='rgb')
        text10.setText(w_10)
        text11.setColor(c_11, colorSpace='rgb')
        text11.setText(w_11)
        text12.setColor(c_12, colorSpace='rgb')
        text12.setText(w_12)
        text13.setColor(c_13, colorSpace='rgb')
        text13.setText(w_13)
        text14.setColor(c_14, colorSpace='rgb')
        text14.setText(w_14)
        text15.setColor(c_15, colorSpace='rgb')
        text15.setText(w_15)
        # store start times for RSVP
        RSVP.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        RSVP.tStart = globalClock.getTime(format='float')
        RSVP.status = STARTED
        thisExp.addData('RSVP.started', RSVP.tStart)
        RSVP.maxDuration = None
        # keep track of which components have finished
        RSVPComponents = RSVP.components
        for thisComponent in RSVP.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RSVP" ---
        thisExp.currentRoutine = RSVP
        RSVP.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.25:
            # if trial has changed, end Routine now
            if hasattr(thisRun1, 'status') and thisRun1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text1* updates
            
            # if text1 is starting this frame...
            if text1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text1.frameNStart = frameN  # exact frame index
                text1.tStart = t  # local t and not account for scr refresh
                text1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text1.started')
                # update status
                text1.status = STARTED
                text1.setAutoDraw(True)
            
            # if text1 is active this frame...
            if text1.status == STARTED:
                # update params
                pass
            
            # if text1 is stopping this frame...
            if text1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text1.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text1.tStop = t  # not accounting for scr refresh
                    text1.tStopRefresh = tThisFlipGlobal  # on global time
                    text1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text1.stopped')
                    # update status
                    text1.status = FINISHED
                    text1.setAutoDraw(False)
            
            # *text2* updates
            
            # if text2 is starting this frame...
            if text2.status == NOT_STARTED and tThisFlip >= 0.15-frameTolerance:
                # keep track of start time/frame for later
                text2.frameNStart = frameN  # exact frame index
                text2.tStart = t  # local t and not account for scr refresh
                text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text2.started')
                # update status
                text2.status = STARTED
                text2.setAutoDraw(True)
            
            # if text2 is active this frame...
            if text2.status == STARTED:
                # update params
                pass
            
            # if text2 is stopping this frame...
            if text2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text2.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text2.tStop = t  # not accounting for scr refresh
                    text2.tStopRefresh = tThisFlipGlobal  # on global time
                    text2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text2.stopped')
                    # update status
                    text2.status = FINISHED
                    text2.setAutoDraw(False)
            
            # *text3* updates
            
            # if text3 is starting this frame...
            if text3.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                text3.frameNStart = frameN  # exact frame index
                text3.tStart = t  # local t and not account for scr refresh
                text3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text3.started')
                # update status
                text3.status = STARTED
                text3.setAutoDraw(True)
            
            # if text3 is active this frame...
            if text3.status == STARTED:
                # update params
                pass
            
            # if text3 is stopping this frame...
            if text3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text3.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text3.tStop = t  # not accounting for scr refresh
                    text3.tStopRefresh = tThisFlipGlobal  # on global time
                    text3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text3.stopped')
                    # update status
                    text3.status = FINISHED
                    text3.setAutoDraw(False)
            
            # *text4* updates
            
            # if text4 is starting this frame...
            if text4.status == NOT_STARTED and tThisFlip >= 0.45-frameTolerance:
                # keep track of start time/frame for later
                text4.frameNStart = frameN  # exact frame index
                text4.tStart = t  # local t and not account for scr refresh
                text4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text4.started')
                # update status
                text4.status = STARTED
                text4.setAutoDraw(True)
            
            # if text4 is active this frame...
            if text4.status == STARTED:
                # update params
                pass
            
            # if text4 is stopping this frame...
            if text4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text4.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text4.tStop = t  # not accounting for scr refresh
                    text4.tStopRefresh = tThisFlipGlobal  # on global time
                    text4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text4.stopped')
                    # update status
                    text4.status = FINISHED
                    text4.setAutoDraw(False)
            
            # *text5* updates
            
            # if text5 is starting this frame...
            if text5.status == NOT_STARTED and tThisFlip >= 0.6-frameTolerance:
                # keep track of start time/frame for later
                text5.frameNStart = frameN  # exact frame index
                text5.tStart = t  # local t and not account for scr refresh
                text5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text5.started')
                # update status
                text5.status = STARTED
                text5.setAutoDraw(True)
            
            # if text5 is active this frame...
            if text5.status == STARTED:
                # update params
                pass
            
            # if text5 is stopping this frame...
            if text5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text5.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text5.tStop = t  # not accounting for scr refresh
                    text5.tStopRefresh = tThisFlipGlobal  # on global time
                    text5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text5.stopped')
                    # update status
                    text5.status = FINISHED
                    text5.setAutoDraw(False)
            
            # *text6* updates
            
            # if text6 is starting this frame...
            if text6.status == NOT_STARTED and tThisFlip >= 0.75-frameTolerance:
                # keep track of start time/frame for later
                text6.frameNStart = frameN  # exact frame index
                text6.tStart = t  # local t and not account for scr refresh
                text6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text6.started')
                # update status
                text6.status = STARTED
                text6.setAutoDraw(True)
            
            # if text6 is active this frame...
            if text6.status == STARTED:
                # update params
                pass
            
            # if text6 is stopping this frame...
            if text6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text6.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text6.tStop = t  # not accounting for scr refresh
                    text6.tStopRefresh = tThisFlipGlobal  # on global time
                    text6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text6.stopped')
                    # update status
                    text6.status = FINISHED
                    text6.setAutoDraw(False)
            
            # *text7* updates
            
            # if text7 is starting this frame...
            if text7.status == NOT_STARTED and tThisFlip >= 0.9-frameTolerance:
                # keep track of start time/frame for later
                text7.frameNStart = frameN  # exact frame index
                text7.tStart = t  # local t and not account for scr refresh
                text7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text7.started')
                # update status
                text7.status = STARTED
                text7.setAutoDraw(True)
            
            # if text7 is active this frame...
            if text7.status == STARTED:
                # update params
                pass
            
            # if text7 is stopping this frame...
            if text7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text7.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text7.tStop = t  # not accounting for scr refresh
                    text7.tStopRefresh = tThisFlipGlobal  # on global time
                    text7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text7.stopped')
                    # update status
                    text7.status = FINISHED
                    text7.setAutoDraw(False)
            
            # *text8* updates
            
            # if text8 is starting this frame...
            if text8.status == NOT_STARTED and tThisFlip >= 1.05-frameTolerance:
                # keep track of start time/frame for later
                text8.frameNStart = frameN  # exact frame index
                text8.tStart = t  # local t and not account for scr refresh
                text8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text8.started')
                # update status
                text8.status = STARTED
                text8.setAutoDraw(True)
            
            # if text8 is active this frame...
            if text8.status == STARTED:
                # update params
                pass
            
            # if text8 is stopping this frame...
            if text8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text8.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text8.tStop = t  # not accounting for scr refresh
                    text8.tStopRefresh = tThisFlipGlobal  # on global time
                    text8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text8.stopped')
                    # update status
                    text8.status = FINISHED
                    text8.setAutoDraw(False)
            
            # *text9* updates
            
            # if text9 is starting this frame...
            if text9.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                text9.frameNStart = frameN  # exact frame index
                text9.tStart = t  # local t and not account for scr refresh
                text9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text9.started')
                # update status
                text9.status = STARTED
                text9.setAutoDraw(True)
            
            # if text9 is active this frame...
            if text9.status == STARTED:
                # update params
                pass
            
            # if text9 is stopping this frame...
            if text9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text9.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text9.tStop = t  # not accounting for scr refresh
                    text9.tStopRefresh = tThisFlipGlobal  # on global time
                    text9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text9.stopped')
                    # update status
                    text9.status = FINISHED
                    text9.setAutoDraw(False)
            
            # *text10* updates
            
            # if text10 is starting this frame...
            if text10.status == NOT_STARTED and tThisFlip >= 1.35-frameTolerance:
                # keep track of start time/frame for later
                text10.frameNStart = frameN  # exact frame index
                text10.tStart = t  # local t and not account for scr refresh
                text10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text10.started')
                # update status
                text10.status = STARTED
                text10.setAutoDraw(True)
            
            # if text10 is active this frame...
            if text10.status == STARTED:
                # update params
                pass
            
            # if text10 is stopping this frame...
            if text10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text10.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text10.tStop = t  # not accounting for scr refresh
                    text10.tStopRefresh = tThisFlipGlobal  # on global time
                    text10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text10.stopped')
                    # update status
                    text10.status = FINISHED
                    text10.setAutoDraw(False)
            
            # *text11* updates
            
            # if text11 is starting this frame...
            if text11.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                text11.frameNStart = frameN  # exact frame index
                text11.tStart = t  # local t and not account for scr refresh
                text11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text11.started')
                # update status
                text11.status = STARTED
                text11.setAutoDraw(True)
            
            # if text11 is active this frame...
            if text11.status == STARTED:
                # update params
                pass
            
            # if text11 is stopping this frame...
            if text11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text11.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text11.tStop = t  # not accounting for scr refresh
                    text11.tStopRefresh = tThisFlipGlobal  # on global time
                    text11.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text11.stopped')
                    # update status
                    text11.status = FINISHED
                    text11.setAutoDraw(False)
            
            # *text12* updates
            
            # if text12 is starting this frame...
            if text12.status == NOT_STARTED and tThisFlip >= 1.65-frameTolerance:
                # keep track of start time/frame for later
                text12.frameNStart = frameN  # exact frame index
                text12.tStart = t  # local t and not account for scr refresh
                text12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text12.started')
                # update status
                text12.status = STARTED
                text12.setAutoDraw(True)
            
            # if text12 is active this frame...
            if text12.status == STARTED:
                # update params
                pass
            
            # if text12 is stopping this frame...
            if text12.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text12.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text12.tStop = t  # not accounting for scr refresh
                    text12.tStopRefresh = tThisFlipGlobal  # on global time
                    text12.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text12.stopped')
                    # update status
                    text12.status = FINISHED
                    text12.setAutoDraw(False)
            
            # *text13* updates
            
            # if text13 is starting this frame...
            if text13.status == NOT_STARTED and tThisFlip >= 1.8-frameTolerance:
                # keep track of start time/frame for later
                text13.frameNStart = frameN  # exact frame index
                text13.tStart = t  # local t and not account for scr refresh
                text13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text13.started')
                # update status
                text13.status = STARTED
                text13.setAutoDraw(True)
            
            # if text13 is active this frame...
            if text13.status == STARTED:
                # update params
                pass
            
            # if text13 is stopping this frame...
            if text13.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text13.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text13.tStop = t  # not accounting for scr refresh
                    text13.tStopRefresh = tThisFlipGlobal  # on global time
                    text13.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text13.stopped')
                    # update status
                    text13.status = FINISHED
                    text13.setAutoDraw(False)
            
            # *text14* updates
            
            # if text14 is starting this frame...
            if text14.status == NOT_STARTED and tThisFlip >= 1.95-frameTolerance:
                # keep track of start time/frame for later
                text14.frameNStart = frameN  # exact frame index
                text14.tStart = t  # local t and not account for scr refresh
                text14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text14.started')
                # update status
                text14.status = STARTED
                text14.setAutoDraw(True)
            
            # if text14 is active this frame...
            if text14.status == STARTED:
                # update params
                pass
            
            # if text14 is stopping this frame...
            if text14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text14.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text14.tStop = t  # not accounting for scr refresh
                    text14.tStopRefresh = tThisFlipGlobal  # on global time
                    text14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text14.stopped')
                    # update status
                    text14.status = FINISHED
                    text14.setAutoDraw(False)
            
            # *text15* updates
            
            # if text15 is starting this frame...
            if text15.status == NOT_STARTED and tThisFlip >= 2.1-frameTolerance:
                # keep track of start time/frame for later
                text15.frameNStart = frameN  # exact frame index
                text15.tStart = t  # local t and not account for scr refresh
                text15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text15.started')
                # update status
                text15.status = STARTED
                text15.setAutoDraw(True)
            
            # if text15 is active this frame...
            if text15.status == STARTED:
                # update params
                pass
            
            # if text15 is stopping this frame...
            if text15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text15.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text15.tStop = t  # not accounting for scr refresh
                    text15.tStopRefresh = tThisFlipGlobal  # on global time
                    text15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text15.stopped')
                    # update status
                    text15.status = FINISHED
                    text15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=RSVP,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                RSVP.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if RSVP.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in RSVP.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RSVP" ---
        for thisComponent in RSVP.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for RSVP
        RSVP.tStop = globalClock.getTime(format='float')
        RSVP.tStopRefresh = tThisFlipGlobal
        thisExp.addData('RSVP.stopped', RSVP.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if RSVP.maxDurationReached:
            routineTimer.addTime(-RSVP.maxDuration)
        elif RSVP.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.250000)
        
        # --- Prepare to start Routine "mcq" ---
        # create an object to store info about Routine mcq
        mcq = data.Routine(
            name='mcq',
            components=[text, mcq1, mcq2, mcq3, mcq4, key_resp],
        )
        mcq.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        mcq1.setText("1. " + str(mcq_option_1))
        mcq2.setText("2. " + str(mcq_option_2))
        mcq3.setText("3. " + str(mcq_option_3))
        mcq4.setText("4. " + str(mcq_option_4))
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for mcq
        mcq.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mcq.tStart = globalClock.getTime(format='float')
        mcq.status = STARTED
        thisExp.addData('mcq.started', mcq.tStart)
        mcq.maxDuration = None
        # keep track of which components have finished
        mcqComponents = mcq.components
        for thisComponent in mcq.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mcq" ---
        thisExp.currentRoutine = mcq
        mcq.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun1, 'status') and thisRun1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *mcq1* updates
            
            # if mcq1 is starting this frame...
            if mcq1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq1.frameNStart = frameN  # exact frame index
                mcq1.tStart = t  # local t and not account for scr refresh
                mcq1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq1.started')
                # update status
                mcq1.status = STARTED
                mcq1.setAutoDraw(True)
            
            # if mcq1 is active this frame...
            if mcq1.status == STARTED:
                # update params
                pass
            
            # *mcq2* updates
            
            # if mcq2 is starting this frame...
            if mcq2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq2.frameNStart = frameN  # exact frame index
                mcq2.tStart = t  # local t and not account for scr refresh
                mcq2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq2.started')
                # update status
                mcq2.status = STARTED
                mcq2.setAutoDraw(True)
            
            # if mcq2 is active this frame...
            if mcq2.status == STARTED:
                # update params
                pass
            
            # *mcq3* updates
            
            # if mcq3 is starting this frame...
            if mcq3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq3.frameNStart = frameN  # exact frame index
                mcq3.tStart = t  # local t and not account for scr refresh
                mcq3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq3.started')
                # update status
                mcq3.status = STARTED
                mcq3.setAutoDraw(True)
            
            # if mcq3 is active this frame...
            if mcq3.status == STARTED:
                # update params
                pass
            
            # *mcq4* updates
            
            # if mcq4 is starting this frame...
            if mcq4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq4.frameNStart = frameN  # exact frame index
                mcq4.tStart = t  # local t and not account for scr refresh
                mcq4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq4.started')
                # update status
                mcq4.status = STARTED
                mcq4.setAutoDraw(True)
            
            # if mcq4 is active this frame...
            if mcq4.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct_key)) or (key_resp.keys == correct_key):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mcq,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mcq.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mcq.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mcq.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mcq" ---
        for thisComponent in mcq.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mcq
        mcq.tStop = globalClock.getTime(format='float')
        mcq.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mcq.stopped', mcq.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for run1 (TrialHandler)
        run1.addData('key_resp.keys',key_resp.keys)
        run1.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            run1.addData('key_resp.rt', key_resp.rt)
            run1.addData('key_resp.duration', key_resp.duration)
        # the Routine "mcq" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRun1 as finished
        if hasattr(thisRun1, 'status'):
            thisRun1.status = FINISHED
        # if awaiting a pause, pause now
        if run1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            run1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'run1'
    run1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_3" ---
    # create an object to store info about Routine break_3
    break_3 = data.Routine(
        name='break_3',
        components=[text_3, key_resp_3],
    )
    break_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for break_3
    break_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_3.tStart = globalClock.getTime(format='float')
    break_3.status = STARTED
    thisExp.addData('break_3.started', break_3.tStart)
    break_3.maxDuration = None
    # keep track of which components have finished
    break_3Components = break_3.components
    for thisComponent in break_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_3" ---
    thisExp.currentRoutine = break_3
    break_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=break_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            break_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if break_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in break_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_3" ---
    for thisComponent in break_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_3
    break_3.tStop = globalClock.getTime(format='float')
    break_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_3.stopped', break_3.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "break_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    run2 = data.TrialHandler2(
        name='run2',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('run12.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(run2)  # add the loop to the experiment
    thisRun2 = run2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun2.rgb)
    if thisRun2 != None:
        for paramName in thisRun2:
            globals()[paramName] = thisRun2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRun2 in run2:
        run2.status = STARTED
        if hasattr(thisRun2, 'status'):
            thisRun2.status = STARTED
        currentLoop = run2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRun2.rgb)
        if thisRun2 != None:
            for paramName in thisRun2:
                globals()[paramName] = thisRun2[paramName]
        
        # --- Prepare to start Routine "Plus" ---
        # create an object to store info about Routine Plus
        Plus = data.Routine(
            name='Plus',
            components=[plus],
        )
        Plus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Plus
        Plus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Plus.tStart = globalClock.getTime(format='float')
        Plus.status = STARTED
        thisExp.addData('Plus.started', Plus.tStart)
        Plus.maxDuration = None
        # keep track of which components have finished
        PlusComponents = Plus.components
        for thisComponent in Plus.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Plus" ---
        thisExp.currentRoutine = Plus
        Plus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisRun2, 'status') and thisRun2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *plus* updates
            
            # if plus is starting this frame...
            if plus.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                plus.frameNStart = frameN  # exact frame index
                plus.tStart = t  # local t and not account for scr refresh
                plus.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(plus, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'plus.started')
                # update status
                plus.status = STARTED
                plus.setAutoDraw(True)
            
            # if plus is active this frame...
            if plus.status == STARTED:
                # update params
                pass
            
            # if plus is stopping this frame...
            if plus.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > plus.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    plus.tStop = t  # not accounting for scr refresh
                    plus.tStopRefresh = tThisFlipGlobal  # on global time
                    plus.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'plus.stopped')
                    # update status
                    plus.status = FINISHED
                    plus.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Plus,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Plus.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Plus.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Plus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Plus" ---
        for thisComponent in Plus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Plus
        Plus.tStop = globalClock.getTime(format='float')
        Plus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Plus.stopped', Plus.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Plus.maxDurationReached:
            routineTimer.addTime(-Plus.maxDuration)
        elif Plus.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "RSVP" ---
        # create an object to store info about Routine RSVP
        RSVP = data.Routine(
            name='RSVP',
            components=[text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15],
        )
        RSVP.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text1.setColor(c_1, colorSpace='rgb')
        text1.setText(w_1
        )
        text2.setColor(c_2, colorSpace='rgb')
        text2.setText(w_2)
        text3.setColor(c_3, colorSpace='rgb')
        text3.setText(w_3)
        text4.setColor(c_4, colorSpace='rgb')
        text4.setText(w_4)
        text5.setColor(c_5, colorSpace='rgb')
        text5.setText(w_5)
        text6.setColor(c_6, colorSpace='rgb')
        text6.setText(w_6)
        text7.setColor(c_7, colorSpace='rgb')
        text7.setText(w_7)
        text8.setColor(c_8, colorSpace='rgb')
        text8.setText(w_8)
        text9.setColor(c_9, colorSpace='rgb')
        text9.setText(w_9)
        text10.setColor(c_10, colorSpace='rgb')
        text10.setText(w_10)
        text11.setColor(c_11, colorSpace='rgb')
        text11.setText(w_11)
        text12.setColor(c_12, colorSpace='rgb')
        text12.setText(w_12)
        text13.setColor(c_13, colorSpace='rgb')
        text13.setText(w_13)
        text14.setColor(c_14, colorSpace='rgb')
        text14.setText(w_14)
        text15.setColor(c_15, colorSpace='rgb')
        text15.setText(w_15)
        # store start times for RSVP
        RSVP.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        RSVP.tStart = globalClock.getTime(format='float')
        RSVP.status = STARTED
        thisExp.addData('RSVP.started', RSVP.tStart)
        RSVP.maxDuration = None
        # keep track of which components have finished
        RSVPComponents = RSVP.components
        for thisComponent in RSVP.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "RSVP" ---
        thisExp.currentRoutine = RSVP
        RSVP.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.25:
            # if trial has changed, end Routine now
            if hasattr(thisRun2, 'status') and thisRun2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text1* updates
            
            # if text1 is starting this frame...
            if text1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text1.frameNStart = frameN  # exact frame index
                text1.tStart = t  # local t and not account for scr refresh
                text1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text1.started')
                # update status
                text1.status = STARTED
                text1.setAutoDraw(True)
            
            # if text1 is active this frame...
            if text1.status == STARTED:
                # update params
                pass
            
            # if text1 is stopping this frame...
            if text1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text1.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text1.tStop = t  # not accounting for scr refresh
                    text1.tStopRefresh = tThisFlipGlobal  # on global time
                    text1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text1.stopped')
                    # update status
                    text1.status = FINISHED
                    text1.setAutoDraw(False)
            
            # *text2* updates
            
            # if text2 is starting this frame...
            if text2.status == NOT_STARTED and tThisFlip >= 0.15-frameTolerance:
                # keep track of start time/frame for later
                text2.frameNStart = frameN  # exact frame index
                text2.tStart = t  # local t and not account for scr refresh
                text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text2.started')
                # update status
                text2.status = STARTED
                text2.setAutoDraw(True)
            
            # if text2 is active this frame...
            if text2.status == STARTED:
                # update params
                pass
            
            # if text2 is stopping this frame...
            if text2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text2.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text2.tStop = t  # not accounting for scr refresh
                    text2.tStopRefresh = tThisFlipGlobal  # on global time
                    text2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text2.stopped')
                    # update status
                    text2.status = FINISHED
                    text2.setAutoDraw(False)
            
            # *text3* updates
            
            # if text3 is starting this frame...
            if text3.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                text3.frameNStart = frameN  # exact frame index
                text3.tStart = t  # local t and not account for scr refresh
                text3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text3.started')
                # update status
                text3.status = STARTED
                text3.setAutoDraw(True)
            
            # if text3 is active this frame...
            if text3.status == STARTED:
                # update params
                pass
            
            # if text3 is stopping this frame...
            if text3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text3.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text3.tStop = t  # not accounting for scr refresh
                    text3.tStopRefresh = tThisFlipGlobal  # on global time
                    text3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text3.stopped')
                    # update status
                    text3.status = FINISHED
                    text3.setAutoDraw(False)
            
            # *text4* updates
            
            # if text4 is starting this frame...
            if text4.status == NOT_STARTED and tThisFlip >= 0.45-frameTolerance:
                # keep track of start time/frame for later
                text4.frameNStart = frameN  # exact frame index
                text4.tStart = t  # local t and not account for scr refresh
                text4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text4.started')
                # update status
                text4.status = STARTED
                text4.setAutoDraw(True)
            
            # if text4 is active this frame...
            if text4.status == STARTED:
                # update params
                pass
            
            # if text4 is stopping this frame...
            if text4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text4.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text4.tStop = t  # not accounting for scr refresh
                    text4.tStopRefresh = tThisFlipGlobal  # on global time
                    text4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text4.stopped')
                    # update status
                    text4.status = FINISHED
                    text4.setAutoDraw(False)
            
            # *text5* updates
            
            # if text5 is starting this frame...
            if text5.status == NOT_STARTED and tThisFlip >= 0.6-frameTolerance:
                # keep track of start time/frame for later
                text5.frameNStart = frameN  # exact frame index
                text5.tStart = t  # local t and not account for scr refresh
                text5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text5.started')
                # update status
                text5.status = STARTED
                text5.setAutoDraw(True)
            
            # if text5 is active this frame...
            if text5.status == STARTED:
                # update params
                pass
            
            # if text5 is stopping this frame...
            if text5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text5.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text5.tStop = t  # not accounting for scr refresh
                    text5.tStopRefresh = tThisFlipGlobal  # on global time
                    text5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text5.stopped')
                    # update status
                    text5.status = FINISHED
                    text5.setAutoDraw(False)
            
            # *text6* updates
            
            # if text6 is starting this frame...
            if text6.status == NOT_STARTED and tThisFlip >= 0.75-frameTolerance:
                # keep track of start time/frame for later
                text6.frameNStart = frameN  # exact frame index
                text6.tStart = t  # local t and not account for scr refresh
                text6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text6.started')
                # update status
                text6.status = STARTED
                text6.setAutoDraw(True)
            
            # if text6 is active this frame...
            if text6.status == STARTED:
                # update params
                pass
            
            # if text6 is stopping this frame...
            if text6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text6.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text6.tStop = t  # not accounting for scr refresh
                    text6.tStopRefresh = tThisFlipGlobal  # on global time
                    text6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text6.stopped')
                    # update status
                    text6.status = FINISHED
                    text6.setAutoDraw(False)
            
            # *text7* updates
            
            # if text7 is starting this frame...
            if text7.status == NOT_STARTED and tThisFlip >= 0.9-frameTolerance:
                # keep track of start time/frame for later
                text7.frameNStart = frameN  # exact frame index
                text7.tStart = t  # local t and not account for scr refresh
                text7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text7.started')
                # update status
                text7.status = STARTED
                text7.setAutoDraw(True)
            
            # if text7 is active this frame...
            if text7.status == STARTED:
                # update params
                pass
            
            # if text7 is stopping this frame...
            if text7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text7.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text7.tStop = t  # not accounting for scr refresh
                    text7.tStopRefresh = tThisFlipGlobal  # on global time
                    text7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text7.stopped')
                    # update status
                    text7.status = FINISHED
                    text7.setAutoDraw(False)
            
            # *text8* updates
            
            # if text8 is starting this frame...
            if text8.status == NOT_STARTED and tThisFlip >= 1.05-frameTolerance:
                # keep track of start time/frame for later
                text8.frameNStart = frameN  # exact frame index
                text8.tStart = t  # local t and not account for scr refresh
                text8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text8.started')
                # update status
                text8.status = STARTED
                text8.setAutoDraw(True)
            
            # if text8 is active this frame...
            if text8.status == STARTED:
                # update params
                pass
            
            # if text8 is stopping this frame...
            if text8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text8.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text8.tStop = t  # not accounting for scr refresh
                    text8.tStopRefresh = tThisFlipGlobal  # on global time
                    text8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text8.stopped')
                    # update status
                    text8.status = FINISHED
                    text8.setAutoDraw(False)
            
            # *text9* updates
            
            # if text9 is starting this frame...
            if text9.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                text9.frameNStart = frameN  # exact frame index
                text9.tStart = t  # local t and not account for scr refresh
                text9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text9.started')
                # update status
                text9.status = STARTED
                text9.setAutoDraw(True)
            
            # if text9 is active this frame...
            if text9.status == STARTED:
                # update params
                pass
            
            # if text9 is stopping this frame...
            if text9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text9.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text9.tStop = t  # not accounting for scr refresh
                    text9.tStopRefresh = tThisFlipGlobal  # on global time
                    text9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text9.stopped')
                    # update status
                    text9.status = FINISHED
                    text9.setAutoDraw(False)
            
            # *text10* updates
            
            # if text10 is starting this frame...
            if text10.status == NOT_STARTED and tThisFlip >= 1.35-frameTolerance:
                # keep track of start time/frame for later
                text10.frameNStart = frameN  # exact frame index
                text10.tStart = t  # local t and not account for scr refresh
                text10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text10.started')
                # update status
                text10.status = STARTED
                text10.setAutoDraw(True)
            
            # if text10 is active this frame...
            if text10.status == STARTED:
                # update params
                pass
            
            # if text10 is stopping this frame...
            if text10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text10.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text10.tStop = t  # not accounting for scr refresh
                    text10.tStopRefresh = tThisFlipGlobal  # on global time
                    text10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text10.stopped')
                    # update status
                    text10.status = FINISHED
                    text10.setAutoDraw(False)
            
            # *text11* updates
            
            # if text11 is starting this frame...
            if text11.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                text11.frameNStart = frameN  # exact frame index
                text11.tStart = t  # local t and not account for scr refresh
                text11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text11.started')
                # update status
                text11.status = STARTED
                text11.setAutoDraw(True)
            
            # if text11 is active this frame...
            if text11.status == STARTED:
                # update params
                pass
            
            # if text11 is stopping this frame...
            if text11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text11.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text11.tStop = t  # not accounting for scr refresh
                    text11.tStopRefresh = tThisFlipGlobal  # on global time
                    text11.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text11.stopped')
                    # update status
                    text11.status = FINISHED
                    text11.setAutoDraw(False)
            
            # *text12* updates
            
            # if text12 is starting this frame...
            if text12.status == NOT_STARTED and tThisFlip >= 1.65-frameTolerance:
                # keep track of start time/frame for later
                text12.frameNStart = frameN  # exact frame index
                text12.tStart = t  # local t and not account for scr refresh
                text12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text12.started')
                # update status
                text12.status = STARTED
                text12.setAutoDraw(True)
            
            # if text12 is active this frame...
            if text12.status == STARTED:
                # update params
                pass
            
            # if text12 is stopping this frame...
            if text12.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text12.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text12.tStop = t  # not accounting for scr refresh
                    text12.tStopRefresh = tThisFlipGlobal  # on global time
                    text12.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text12.stopped')
                    # update status
                    text12.status = FINISHED
                    text12.setAutoDraw(False)
            
            # *text13* updates
            
            # if text13 is starting this frame...
            if text13.status == NOT_STARTED and tThisFlip >= 1.8-frameTolerance:
                # keep track of start time/frame for later
                text13.frameNStart = frameN  # exact frame index
                text13.tStart = t  # local t and not account for scr refresh
                text13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text13.started')
                # update status
                text13.status = STARTED
                text13.setAutoDraw(True)
            
            # if text13 is active this frame...
            if text13.status == STARTED:
                # update params
                pass
            
            # if text13 is stopping this frame...
            if text13.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text13.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text13.tStop = t  # not accounting for scr refresh
                    text13.tStopRefresh = tThisFlipGlobal  # on global time
                    text13.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text13.stopped')
                    # update status
                    text13.status = FINISHED
                    text13.setAutoDraw(False)
            
            # *text14* updates
            
            # if text14 is starting this frame...
            if text14.status == NOT_STARTED and tThisFlip >= 1.95-frameTolerance:
                # keep track of start time/frame for later
                text14.frameNStart = frameN  # exact frame index
                text14.tStart = t  # local t and not account for scr refresh
                text14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text14.started')
                # update status
                text14.status = STARTED
                text14.setAutoDraw(True)
            
            # if text14 is active this frame...
            if text14.status == STARTED:
                # update params
                pass
            
            # if text14 is stopping this frame...
            if text14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text14.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text14.tStop = t  # not accounting for scr refresh
                    text14.tStopRefresh = tThisFlipGlobal  # on global time
                    text14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text14.stopped')
                    # update status
                    text14.status = FINISHED
                    text14.setAutoDraw(False)
            
            # *text15* updates
            
            # if text15 is starting this frame...
            if text15.status == NOT_STARTED and tThisFlip >= 2.1-frameTolerance:
                # keep track of start time/frame for later
                text15.frameNStart = frameN  # exact frame index
                text15.tStart = t  # local t and not account for scr refresh
                text15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text15.started')
                # update status
                text15.status = STARTED
                text15.setAutoDraw(True)
            
            # if text15 is active this frame...
            if text15.status == STARTED:
                # update params
                pass
            
            # if text15 is stopping this frame...
            if text15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text15.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    text15.tStop = t  # not accounting for scr refresh
                    text15.tStopRefresh = tThisFlipGlobal  # on global time
                    text15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text15.stopped')
                    # update status
                    text15.status = FINISHED
                    text15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=RSVP,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                RSVP.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if RSVP.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in RSVP.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "RSVP" ---
        for thisComponent in RSVP.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for RSVP
        RSVP.tStop = globalClock.getTime(format='float')
        RSVP.tStopRefresh = tThisFlipGlobal
        thisExp.addData('RSVP.stopped', RSVP.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if RSVP.maxDurationReached:
            routineTimer.addTime(-RSVP.maxDuration)
        elif RSVP.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.250000)
        
        # --- Prepare to start Routine "mcq" ---
        # create an object to store info about Routine mcq
        mcq = data.Routine(
            name='mcq',
            components=[text, mcq1, mcq2, mcq3, mcq4, key_resp],
        )
        mcq.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        mcq1.setText("1. " + str(mcq_option_1))
        mcq2.setText("2. " + str(mcq_option_2))
        mcq3.setText("3. " + str(mcq_option_3))
        mcq4.setText("4. " + str(mcq_option_4))
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for mcq
        mcq.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mcq.tStart = globalClock.getTime(format='float')
        mcq.status = STARTED
        thisExp.addData('mcq.started', mcq.tStart)
        mcq.maxDuration = None
        # keep track of which components have finished
        mcqComponents = mcq.components
        for thisComponent in mcq.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mcq" ---
        thisExp.currentRoutine = mcq
        mcq.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun2, 'status') and thisRun2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *mcq1* updates
            
            # if mcq1 is starting this frame...
            if mcq1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq1.frameNStart = frameN  # exact frame index
                mcq1.tStart = t  # local t and not account for scr refresh
                mcq1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq1.started')
                # update status
                mcq1.status = STARTED
                mcq1.setAutoDraw(True)
            
            # if mcq1 is active this frame...
            if mcq1.status == STARTED:
                # update params
                pass
            
            # *mcq2* updates
            
            # if mcq2 is starting this frame...
            if mcq2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq2.frameNStart = frameN  # exact frame index
                mcq2.tStart = t  # local t and not account for scr refresh
                mcq2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq2.started')
                # update status
                mcq2.status = STARTED
                mcq2.setAutoDraw(True)
            
            # if mcq2 is active this frame...
            if mcq2.status == STARTED:
                # update params
                pass
            
            # *mcq3* updates
            
            # if mcq3 is starting this frame...
            if mcq3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq3.frameNStart = frameN  # exact frame index
                mcq3.tStart = t  # local t and not account for scr refresh
                mcq3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq3.started')
                # update status
                mcq3.status = STARTED
                mcq3.setAutoDraw(True)
            
            # if mcq3 is active this frame...
            if mcq3.status == STARTED:
                # update params
                pass
            
            # *mcq4* updates
            
            # if mcq4 is starting this frame...
            if mcq4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                mcq4.frameNStart = frameN  # exact frame index
                mcq4.tStart = t  # local t and not account for scr refresh
                mcq4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mcq4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mcq4.started')
                # update status
                mcq4.status = STARTED
                mcq4.setAutoDraw(True)
            
            # if mcq4 is active this frame...
            if mcq4.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(correct_key)) or (key_resp.keys == correct_key):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=mcq,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                mcq.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if mcq.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in mcq.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mcq" ---
        for thisComponent in mcq.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mcq
        mcq.tStop = globalClock.getTime(format='float')
        mcq.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mcq.stopped', mcq.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for run2 (TrialHandler)
        run2.addData('key_resp.keys',key_resp.keys)
        run2.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            run2.addData('key_resp.rt', key_resp.rt)
            run2.addData('key_resp.duration', key_resp.duration)
        # the Routine "mcq" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRun2 as finished
        if hasattr(thisRun2, 'status'):
            thisRun2.status = FINISHED
        # if awaiting a pause, pause now
        if run2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            run2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'run2'
    run2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "finish" ---
    # create an object to store info about Routine finish
    finish = data.Routine(
        name='finish',
        components=[text_4],
    )
    finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for finish
    finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    finish.tStart = globalClock.getTime(format='float')
    finish.status = STARTED
    thisExp.addData('finish.started', finish.tStart)
    finish.maxDuration = None
    # keep track of which components have finished
    finishComponents = finish.components
    for thisComponent in finish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "finish" ---
    thisExp.currentRoutine = finish
    finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.tStopRefresh = tThisFlipGlobal  # on global time
                text_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.stopped')
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=finish,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            finish.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if finish.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "finish" ---
    for thisComponent in finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for finish
    finish.tStop = globalClock.getTime(format='float')
    finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('finish.stopped', finish.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if finish.maxDurationReached:
        routineTimer.addTime(-finish.maxDuration)
    elif finish.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
