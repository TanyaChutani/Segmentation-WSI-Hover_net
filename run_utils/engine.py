
import tqdm
from enum import Enum

from .callbacks import *

class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STARTED = "started"
    COMPLETED = "completed"
    EXCEPTION_RAISED = "exception_raised"
####

class State(object):
    """
    An object that is used to pass internal and 
    user-defined state between event handlers
    """
    def __init__(self):
        # settings propagated from config
        self.logging = None
        self.log_dir = None
        self.log_info = None


        # internal variable
        self.curr_epoch_step   = 0 # current step in epoch
        self.curr_global_step  = 0 # current global step
        self.curr_epoch = 0 # current global epoch

        # TODO: [LOW] better document this
        # for outputing value that will be tracked per step
        # "scalar" will always be printed out and added to the tensorboard
        # "images" will need dedicated function to process and added to the tensorboard 

        # ! naming should match with types supported for serialize
        # TODO: Need way to dynamically adding new types
        self.tracked_step_output = {
            'scalar' : {}, # type : {variable_name : variablee_value}
            'image'  : {},
            'conf_mat' : {},
        }
        # TODO: find way to known which method bind/interact with which value

        self.epoch_accumulated_output = {} # all output of the current epoch

        # TODO: soft reset for pertain variable for N epochs
        self.run_accumulated_output = [] # of run until reseted

        # * depend on the type of training i.e GAN, the updated accumulated may be different
        self.step_raw_output  = None # holder for output returned after current runstep

        self.global_state = None
        return

    def reset_variable(self):
        self.tracked_step_output = {
            'scalar' : {}, # type : {variable_name : variable_value}
            'image'  : {},
            'conf_mat' : {},
        }

        # TODO: [CRITICAL] refactor this
        if self.curr_epoch % self.pertain_n_epoch_output == 0:
            self.run_accumulated_output = []

        self.epoch_accumulated_output = {}

        # * depend on the type of training i.e GAN, the updated accumulated may be different
        self.step_raw_output  = None # holder for output returned after current runstep
        return
####
class RunEngine(object):
    def __init__(self,
                engine_name=None,
                dataloader=None,
                run_step=None, 
                run_info=None,
                log_info=None, # TODO: refactor this with trainer.py
            ):

        # * auto set all input as object variables
        self.engine_name = engine_name
        self.run_step = run_step
        self.dataloader = dataloader		

        # * global variable/object holder shared between all event handler
        self.state = State()
        # * check if correctly referenced, not new copies
        self.state.attached_engine_name = engine_name # TODO: redundant?
        self.state.run_info = run_info 
        self.state.log_info = log_info
        self.state.batch_size = dataloader.batch_size

        # TODO: [CRITICAL] match all the mechanism outline with opt
        self.state.pertain_n_epoch_output = 1 if engine_name == 'valid' else 1

        self.event_handler_dict = { event : [] for event in Events }

        # TODO: think about this more
        # to share global state across a chain of RunEngine such as 
        # from the engine for training to engine for validation

        #
        self.terminate = False
        return

    def __reset_state(self):
        # TODO: think about this more, look too redundant
        new_state = State()
        new_state.attached_engine_name = self.state.attached_engine_name
        new_state.run_info = self.state.run_info 
        new_state.log_info = self.state.log_info
        self.state = new_state
        return 

    def __trigger_events(self, event):        
        for callback in self.event_handler_dict[event]:
            callback.run(self.state, event) 
            # TODO: exception and throwing error with name or sthg to allow trace back
        return

    # TODO: variable to indicate output dependency between handler !
    def add_event_handler(self, event_name, handler):
        self.event_handler_dict[event_name].append(handler)

    ####
    def run(self, nr_epoch=1, shared_state=None, chained=False):
        
        # TODO: refactor this
        if chained: 
            self.state.curr_epoch = 0
        self.state.global_state = shared_state

        while self.state.curr_epoch < nr_epoch:
            self.state.reset_variable() # * reset all EMA holder per epoch

            if not chained:
                print('----------------EPOCH %d' % (self.state.curr_epoch+1))

            self.__trigger_events(Events.EPOCH_STARTED)

            pbar_format = 'Processing: |{bar}| '\
                          '{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
            if self.engine_name == 'train':
                pbar_format += 'Batch_Loss = {postfix[1][Batch_Loss]:0.5f}|'\
                               'EMA_Loss = {postfix[1][EMA_Loss]:0.5f}'
                pbar = tqdm.tqdm(total=len(self.dataloader), leave=True,
                            bar_format=pbar_format, ascii='#',
                            postfix=['', dict(Batch_Loss=float('NaN'), 
                                              EMA_Loss=float('NaN'))])
            else:
                pbar = tqdm.tqdm(total=len(self.dataloader), leave=True,
                            bar_format=pbar_format, ascii='#')

            for data_batch in self.dataloader:
                self.__trigger_events(Events.STEP_STARTED)

                step_output = self.run_step(data_batch, self.state.run_info)
                self.state.step_raw_output = step_output

                self.__trigger_events(Events.STEP_COMPLETED)
                self.state.curr_global_step += 1
                self.state.curr_epoch_step += 1

                if self.engine_name == 'train':
                    pbar.postfix[1]["Batch_Loss"] = step_output['EMA']['loss']
                    pbar.postfix[1]["EMA_Loss"] = self.state.tracked_step_output['scalar']['loss']
                pbar.update()
            pbar.close() # to flush out the bar before doing end of epoch reporting
            self.state.curr_epoch += 1
            self.__trigger_events(Events.EPOCH_COMPLETED)

            # TODO: [CRITICAL] align the protocol
            self.state.run_accumulated_output.append(
                    self.state.epoch_accumulated_output)

        return