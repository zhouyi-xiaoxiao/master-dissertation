"""Run a list of the note's scripts inside ONE interpreter (the start-up of Python + numpy/scipy/mpmath/flint is very slow on the
shared machine).  Each script is executed with runpy as __main__; stdout goes to logs/NN.log, stderr and timing to logs/NN.err;
progress and exit status to logs/run_all_progress.log.  mpmath precision and the Arb precision are reset before every script.

Usage:  python scripts/run_all.py 06 14 16 09 11 13 20 21        (prefixes of the script names; default: the full ordered list)
Order constraints: 06 -> 14 -> 16 -> 09 -> 11;  13 -> 18;  19, 11 and the final tex -> 21;  10 reads 14;  17 reads 06, 16;  20 reads 06.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, glob, time, runpy, io, contextlib, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = _os.path.join(_R, 'out', 'logs', 'msc_rigorous_spectral')
DEFAULT = ["06", "14", "16", "09", "11", "13", "18", "19", "20", "21", "01", "02", "03", "04", "05", "07", "08", "10", "12", "15", "17"]
todo = sys.argv[1:] or DEFAULT
prog = open(_os.path.join(_R, 'out', 'logs', 'msc_rigorous_spectral', 'run_all_progress.log'), "a")
def say(msg):
    prog.write(time.strftime("%H:%M:%S ") + msg + "\n"); prog.flush()
say("START " + " ".join(todo))
bad = []
for pre in todo:
    hits = sorted(glob.glob(os.path.join(HERE, pre + "_*.py")))
    assert len(hits) == 1, (pre, hits)
    path = hits[0]
    import mpmath, flint
    mpmath.mp.dps = 15; flint.ctx.prec = 53
    out = open(os.path.join(LOGS, pre + ".log"), "w"); err = open(os.path.join(LOGS, pre + ".err"), "w")
    t0 = time.time(); c0 = time.process_time(); status = 0
    argv = sys.argv; sys.argv = [path]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        status = 99; err.write(traceback.format_exc())
    sys.argv = argv
    err.write(f"\nexit status {status}; wall {time.time()-t0:.1f} s; cpu {time.process_time()-c0:.1f} s (run_all.py)\n")
    out.close(); err.close()
    last = open(os.path.join(LOGS, pre + ".log")).read().strip().splitlines()[-1:] or [""]
    say(f"{os.path.basename(path)}: exit {status}, wall {time.time()-t0:.0f} s, cpu {time.process_time()-c0:.0f} s; last line: {last[0][:150]}")
    if status != 0: bad.append(pre)
say("DONE; failures: " + (" ".join(bad) or "none"))
sys.exit(1 if bad else 0)
