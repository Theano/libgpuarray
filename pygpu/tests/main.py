import os

from nose.config import Config
from nose.plugins.manager import PluginManager
from numpy.testing.nosetester import import_nose, NoseTester
from numpy.testing.noseclasses import KnownFailure, NumpyTestProgram
import nose.plugins.builtin


class NoseTester(NoseTester):
    """
    Nose test runner.

    This class enables running nose tests from inside libgpuarray,
    by calling pygpu.test().
    This version is more adapted to what we want than Numpy's one.
    """

    def _test_argv(self, verbose, extra_argv):
        """
        Generate argv for nosetest command

        Parameters
        ----------
        verbose: int
            Verbosity value for test outputs, in the range 1-10.
            Default is 1.
        extra_argv: list
            List with any extra arguments to pass to nosetests.

        """
        # self.package_path = os.path.abspath(self.package_path)
        argv = [__file__, self.package_path]
        argv += ['--verbosity', str(verbose)]
        if extra_argv:
            argv += extra_argv
        return argv

    def _show_system_info(self):
        import pygpu
        # print ("pygpu version %s" % pygpu.__version__)
        pygpu_dir = os.path.dirname(pygpu.__file__)
        print("pygpu is installed in %s" % pygpu_dir)

        super(NoseTester, self)._show_system_info()

    def prepare_test_args(self, verbose=1, extra_argv=None, coverage=False,
                          capture=True, knownfailure=True):
        """
        Prepare arguments for the `test` method.

        Takes the same arguments as `test`.
        """
        # fail with nice error message if nose is not present
        nose = import_nose()

        # compile argv
        argv = self._test_argv(verbose, extra_argv)

        # numpy way of doing coverage
        if coverage:
            argv += ['--cover-package=%s' % self.package_name,
                     '--with-coverage', '--cover-tests', '--cover-inclusive',
                     '--cover-erase']

        # Capture output only if needed
        if not capture:
            argv += ['-s']

        # construct list of plugins
        plugins = []
        if knownfailure:
            plugins.append(KnownFailure())
        plugins += [p() for p in nose.plugins.builtin.plugins]

        return argv, plugins

    def test(self, verbose=1, extra_argv=None, coverage=False, capture=True,
             knownfailure=True):
        """
        Run tests for module using nose.

        Parameters
        ----------
        verbose: int
            Verbosity value for test outputs, in the range 1-10.
            Default is 1.
        extra_argv: list
            List with any extra arguments to pass to nosetests.
        coverage: bool
            If True, report coverage of pygpu code. Default is False.
        capture: bool
            If True, capture the standard output of the tests, like
            nosetests does in command-line. The output of failing
            tests will be displayed at the end. Default is True.
        knownfailure: bool
            If True, tests raising KnownFailureTest will not be
            considered Errors nor Failure, but reported as "known
            failures" and treated quite like skipped tests.  Default
            is True.

        Returns
        -------
        nose.result.TextTestResult
            The result of running the tests

        """
        # cap verbosity at 3 because nose becomes *very* verbose beyond that
        verbose = min(verbose, 3)
        self._show_system_info()

        cwd = os.getcwd()
        if self.package_path in os.listdir(cwd):
            # The tests give weird errors if the package to test is
            # in current directory.
            raise RuntimeError((
                "This function does not run correctly when, at the time "
                "pygpu was imported, the working directory was pygpu's "
                "parent directory. You should exit your Python prompt, change "
                "directory, then launch Python again, import pygpu, then "
                "launch pygpu.test()."))

        argv, plugins = self.prepare_test_args(verbose, extra_argv, coverage,
                                               capture, knownfailure)

        # The "plugins" keyword of NumpyTestProgram gets ignored if config is
        # specified. Moreover, using "addplugins" instead can lead to strange
        # errors. So, we specify the plugins in the Config as well.
        cfg = Config(includeExe=True, plugins=PluginManager(plugins=plugins))
        t = NumpyTestProgram(argv=argv, exit=False, config=cfg)
        return t.result
