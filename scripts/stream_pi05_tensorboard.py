"""Mirror complete per-step telemetry to promptly flushed TensorBoard events."""
import argparse
import json
from pathlib import Path
import time


def main():
    from torch.utils.tensorboard import SummaryWriter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('/data/tether/runs/pi05-snapflow-10k'))
    args=parser.parse_args()
    writer=SummaryWriter(str(args.run/'tensorboard-live'),purge_step=0,flush_secs=5)
    try:
        with (args.run/'telemetry.jsonl').open() as stream:
            while True:
                position=stream.tell()
                line=stream.readline()
                if not line.endswith('\n'):
                    stream.seek(position)
                    writer.flush()
                    time.sleep(1)
                    continue
                row=json.loads(line)
                for key,value in row.items():
                    if key!='step' and isinstance(value,(int,float)):
                        writer.add_scalar(key,value,row['step'])
    finally:
        writer.close()


if __name__=='__main__':main()
