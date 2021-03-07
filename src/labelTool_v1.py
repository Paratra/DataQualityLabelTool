'''
Author: Ming Song
In order to label time series data
ming.song@uga.edu
CopyRight@2021 MING
'''

import wx
from wx.lib.plot import PolyLine, PlotCanvas, PlotGraphics
import numpy as np
import pickle
import os
# import logging

def drawBarGraph(each_data):

    plot_index = np.arange(0,each_data.shape[0],1)
    raw_wave = PolyLine(np.concatenate((plot_index[:,np.newaxis], each_data[:,np.newaxis]),1), colour='green', legend='Raw_wave', width=3)

    return PlotGraphics([raw_wave])

class LabelFrame(wx.Frame):
    """

    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(LabelFrame, self).__init__(*args, **kw)
        wx.Frame.__init__(self, None, -1, u'Raw_Data', size=wx.Size(1024,768))

        # create a panel in the frame
        self.p = wx.Panel(self, wx.ID_ANY)
        # self.p.SetBackgroundColour('black')

        # create some sizers
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        checkSizer = wx.BoxSizer(wx.HORIZONTAL)

        # create a menu bar
        self.makeMenuBar()

        # for each_data in data:
        self.canvas = PlotCanvas(self.p)
        self.canvas.SetBackgroundColour('black')

        toggleGrid = wx.CheckBox(self.p, label="Show Grid")
        toggleGrid.Bind(wx.EVT_CHECKBOX, self.onToggleGrid)
        toggleLegend = wx.CheckBox(self.p, label="Show Legend")
        toggleLegend.Bind(wx.EVT_CHECKBOX, self.onToggleLegend)


        ##### Buttons
        self.button_good = wx.Button(self.p,
            label="Good-1", pos=(50, 1))
        self.Bind(wx.EVT_BUTTON, self.OnButtonClick_GOOD,
            self.button_good)

        self.button_bad = wx.Button(self.p,
            label="Bad-2", pos=(140, 1))
        self.Bind(wx.EVT_BUTTON, self.OnButtonClick_BAD,
            self.button_bad)

        self.button_go_back = wx.Button(self.p,
            label="Go Back-3", pos=(230, 1))
        self.Bind(wx.EVT_BUTTON, self.OnButtonClick_GO_BACK,
            self.button_go_back)

        self.button_save = wx.Button(self.p,
            label="Save-4", pos=(320, 1))
        self.Bind(wx.EVT_BUTTON, self.OnButtonClick_SAVE,
            self.button_save)



        self.p.SetFocus()
        self.p.Bind(wx.EVT_KEY_DOWN, self.KeyDown)


        # layout the widgets
        mainSizer.Add(self.canvas, 1, wx.EXPAND)
        checkSizer.Add(toggleGrid, 0, wx.ALL, 5)
        checkSizer.Add(toggleLegend, 0, wx.ALL, 5)
        mainSizer.Add(checkSizer)
        self.p.SetSizer(mainSizer)

        # and a status bar
        self.CreateStatusBar()
        self.SetStatusText("CopyRight@2021 Ming Song")

    # def buttonDefine(self):





    #----------------------------------------------------------------------
    def onToggleGrid(self, event):
        """"""
        self.canvas.SetEnableGrid(event.IsChecked())

    #----------------------------------------------------------------------
    def onToggleLegend(self, event):
        """"""
        self.canvas.SetEnableLegend(event.IsChecked())

    #----------------------------------------------------------------------
    def OnButtonClick_GOOD(self, event):
        self.label_result.append(np.concatenate((self.data[self.ind,:], np.array([0]))))
        # import pdb; pdb.set_trace()
        self.ind += 1
        # self.p.SetBackgroundColour('Green')
        self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
        self.canvas.Refresh()
        self.p.Refresh()

    #----------------------------------------------------------------------
    def OnButtonClick_BAD(self, event):
        self.label_result.append(np.concatenate((self.data[self.ind,:], np.array([1]))))
        self.ind += 1
        # self.p.SetBackgroundColour('Green')
        self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
        self.canvas.Refresh()
        self.p.Refresh()

    #----------------------------------------------------------------------
    def OnButtonClick_SAVE(self, event):
        '''
        Need to save:
            # self.ind
            # self.current_file_name
            # self.label_result
        '''

        if not os.path.exists('../data'):
            os.makedirs('../data')
        try:
            np.save(f'../data/labled_{self.current_file_name}',np.asarray(self.label_result))
            with open(f'../data/{self.current_file_name}_info.pk',"wb+") as f:
                pickle.dump([self.ind,self.current_file_name],f)
            print('Labeled data saved!')
        except:
            print('Save failed!')


    #----------------------------------------------------------------------
    def OnButtonClick_GO_BACK(self, event):
        try:
            del self.label_result[-1]
            self.ind -= 1
            # self.p.SetBackgroundColour('Green')
            self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
            self.canvas.Refresh()
            self.p.Refresh()
        except:
            print('This is the first data!')



    #----------------------------------------------------------------------
    def KeyDown_GOOD(self):
        self.label_result.append(np.concatenate((self.data[self.ind,:], np.array([0]))))
        # import pdb; pdb.set_trace()
        self.ind += 1
        # self.p.SetBackgroundColour('Green')
        self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
        self.canvas.Refresh()
        self.p.Refresh()


    #----------------------------------------------------------------------
    def KeyDown_BAD(self):
        self.label_result.append(np.concatenate((self.data[self.ind,:], np.array([1]))))
        self.ind += 1
        # self.p.SetBackgroundColour('Green')
        self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
        self.canvas.Refresh()
        self.p.Refresh()
    #----------------------------------------------------------------------
    def KeyDown_SAVE(self):
        '''
        Need to save:
            # self.ind
            # self.current_file_name
            # self.label_result
        '''

        if not os.path.exists('../data'):
            os.makedirs('../data')
        try:
            np.save(f'../data/labled_{self.current_file_name}',np.asarray(self.label_result))
            with open(f'../data/{self.current_file_name}_info.pk',"wb+") as f:
                pickle.dump([self.ind,self.current_file_name],f)
            print('Labeled data saved!')
        except:
            print('Save failed!')
    #----------------------------------------------------------------------
    def KeyDown_GO_BACK(self):
        try:
            del self.label_result[-1]
            self.ind -= 1
            # self.p.SetBackgroundColour('Green')
            self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))
            self.canvas.Refresh()
            self.p.Refresh()
        except:
            print('This is the first data!')




    #----------------------------------------------------------------------
    def OnLeaveWindow(self, event):
        self.button_good.SetLabel("Good-1")
        # self.button_good.SetLabel("Good-0")
        event.Skip()

    #----------------------------------------------------------------------
    def OnLeaveWindow(self, event):
        self.button_bad.SetLabel("Bad-2")
        event.Skip()

    #----------------------------------------------------------------------
    def OnLeaveWindow(self, event):
        self.button_go_back.SetLabel("Go Back-3")
        event.Skip()

    #----------------------------------------------------------------------
    def OnLeaveWindow(self, event):
        self.button_save.SetLabel("Save-4")
        event.Skip()

    def KeyDown(self, event=None):
        # print("OnKeyDown event %s" % (event.GetKeyCode()))
        # logging.warning("OnKeyDown event %s" % (event))
        self.key_code = event.GetKeyCode()
        # print(type(self.key_code))
        # print(self.key_code)
        if (self.key_code == 49) or (self.key_code == 325): # key 1
            self.KeyDown_GOOD()
            print('Good data!')
        elif (self.key_code == 50) or (self.key_code == 326): # key 2
            self.KeyDown_BAD()
            print('Bad data!')
        elif (self.key_code == 51) or (self.key_code == 327): # key 3
            self.KeyDown_GO_BACK()
            print('Go Back!')
        elif (self.key_code == 52) or (self.key_code == 328): # key 4
            self.KeyDown_SAVE()
            # print('Save!')

    def __OpenSingleFile(self, event):
        filesFilter = "Dicom (*.dcm)|*.dcm|" "All files (*.*)|*.*"
        fileDialog = wx.FileDialog(self, message ="Select File", wildcard = filesFilter, style = wx.FD_OPEN)
        dialogResult = fileDialog.ShowModal()
        if dialogResult !=  wx.ID_OK:
            print('Failed to open file dialog!')
            exit()
        # self.path = fileDialog.GetPath()
        # self.data = np.load(self.path)
        # self.ind = 0
        # self.current_file_name = self.path.split('/')[-1]
        self.current_file_name = 'none.none'
        NPY_FILE = False
        while not NPY_FILE:
            if self.current_file_name.split('.')[1] != 'npy':
                print('Please load a npy file!')
                self.path = fileDialog.GetPath()
                self.data = np.load(self.path)
                self.ind = 0
                self.current_file_name = self.path.split('/')[-1]
            elif self.current_file_name.split('.')[1] == 'npy':
                NPY_FILE = True
                self.label_result = []
        print(f'{self.current_file_name} loaded!')

        # import pdb; pdb.set_trace()
        self.canvas.Draw(drawBarGraph(self.data[self.ind,:1000]/max(self.data[self.ind,:1000])), yAxis=(-3,4))

        # import pdb; pdb.set_trace()
        # self.__TextBox.SetLabel(path)



    def makeMenuBar(self):
        """
        A menu bar is composed of menus, which are composed of menu items.
        This method builds a set of menus and binds handlers to be called
        when the menu item is selected.
        """

        # Make a file menu with Hello and Exit items
        fileMenu = wx.Menu()
        # The "\t..." syntax defines an accelerator key that also triggers
        # the same event
        fileItem = fileMenu.Append(-1, "Open File")
        fileMenu.AppendSeparator()
        self.Bind(wx.EVT_MENU, self.__OpenSingleFile, fileItem)
        # When using a stock ID we don't need to specify the menu item's
        # label
        exitItem = fileMenu.Append(wx.ID_EXIT)


        # Now a help menu for the about item
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        # Make the menu bar and add the two menus to it. The '&' defines
        # that the next letter is the "mnemonic" for the menu item. On the
        # platforms that support it those letters are underlined and can be
        # triggered from the keyboard.
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")
        # menuBar.Append(menu, '&Plot')

        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        # Finally, associate a handler function with the EVT_MENU event for
        # each of the menu items. That means that when that menu item is
        # activated then the associated handler function will be called.

        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
        # self.Bind(wx.EVT_MENU,self.OnPlotDraw1, id=100)


    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)
        # exit()


    def OnHello(self, event):
        """Say hello to the user."""
        wx.MessageBox("Hello again from wxPython")


    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK|wx.ICON_INFORMATION)




def main():
    # When this module is run (not imported) then create the app, the
    # frame, show it, and start the event loop.
    app = wx.App()
    frm = LabelFrame(parent=None, id=-1)
    # frm.start()
    frm.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
