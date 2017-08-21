"""Generic wrapper for read-eval-print-loops, a.k.a. interactive shells
   modified original pexpect(https://github.com/pexpect/pexpect/blob/master/pexpect/replwrap.py) 
   modified for evaluating commands in NCAR Command Language Shell
"""
import os.path
import signal
import sys

import pexpect

PY3 = (sys.version_info[0] >= 3)

if PY3:
    basestring = str

PEXPECT_PROMPT = u'[PEXPECT_PROMPT>'
PEXPECT_CONTINUATION_PROMPT = u'[PEXPECT_PROMPT+'

class REPLWrapper(object):
    """Wrapper for a REPL.
    :param cmd_or_spawn: This can either be an instance of :class:`pexpect.spawn`
      in which a REPL has already been started, or a str command to start a new
      REPL process.
    :param str orig_prompt: The prompt to expect at first.
    :param str prompt_change: A command to change the prompt to something more
      unique. If this is ``None``, the prompt will not be changed. This will
      be formatted with the new and continuation prompts as positional
      parameters, so you can use ``{}`` style formatting to insert them into
      the command.
    :param str new_prompt: The more unique prompt to expect after the change.
    :param str extra_init_cmd: Commands to do extra initialisation, such as
      disabling pagers.
    """
    def __init__(self, cmd_or_spawn, orig_prompt, prompt_change,
                 new_prompt=PEXPECT_PROMPT,
                 continuation_prompt=PEXPECT_CONTINUATION_PROMPT,
                 extra_init_cmd=None,line_output_callback=None):
        self.line_output_callback = line_output_callback
        
        if isinstance(cmd_or_spawn, basestring):
            self.child = pexpect.spawn(cmd_or_spawn, echo=False) #encoding = utf was not working
        else:
            self.child = cmd_or_spawn
        if self.child.echo:
            # Existing spawn instance has echo enabled, disable it
            # to prevent our input from being repeated to output.
            self.child.setecho(False)
            self.child.waitnoecho()

        if prompt_change is None:
            self.prompt = orig_prompt
        else:
            self.set_prompt(orig_prompt,
                        prompt_change.format(new_prompt, continuation_prompt))
            self.prompt = new_prompt
        self.continuation_prompt = continuation_prompt

        self._expect_prompt()

        if extra_init_cmd is not None:
            self.run_command(extra_init_cmd)

    def set_prompt(self, orig_prompt, prompt_change):
        self.child.expect(orig_prompt)
        self.child.sendline(prompt_change)

    def _expect_prompt(self, timeout=-1):
        print( self.prompt )
        return self.child.expect(self.prompt) #had to be simplified

    def run_command(self, command, timeout=-1):
        """Send a command to the REPL, wait for and return output.
        :param str command: The command to send. Trailing newlines are not needed.
          This should be a complete block of input that will trigger execution;
          if a continuation prompt is found after sending input, :exc:`ValueError`
          will be raised.
        :param int timeout: How long to wait for the next prompt. -1 means the
          default from the :class:`pexpect.spawn` object (default 30 seconds).
          None means to wait indefinitely.
        """
        # Split up multiline commands and feed them in bit-by-bit
        cmdlines = command.splitlines()
        # splitlines ignores trailing newlines - add it back in manually
        if command.endswith('\n'):
            cmdlines.append('')
        if not cmdlines:
            raise ValueError("No command was given")

        res = []
        self.child.sendline(cmdlines[0])
        for line in cmdlines[1:]:
            self._expect_prompt(timeout=timeout)
            output = self.child.before.decode().splitlines()
            self.line_output_callback(output)
            self.child.sendline(line)

        # Command was fully submitted, now wait for the next prompt
        if self._expect_prompt(timeout=timeout) == 1:
            # We got the continuation prompt - command was incomplete
            self.child.kill(signal.SIGINT)
            self._expect_prompt(timeout=1)
            raise ValueError("Continuation prompt found - input was incomplete:\n"
                             + command)
        output = self.child.before.decode().splitlines()
        self.line_output_callback(output)
