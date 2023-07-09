from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from typing import Dict, TypedDict, Union, Optional
from collections import OrderedDict

"""Warning: as noted on [Python Documentation of XML module](https://docs.python.org/3/library/xml.html#xml-vulnerabilities), the script is vulnerable to malicious data. Use with caution.
"""


def chk_file(f: Path):
    if not f.is_file():
        raise FileNotFoundError(f"{f.absolute()} is not found.")


def parse_segments(seg_file: Path):
    chk_file(seg_file)

    with open(seg_file) as f:
        root = ET.parse(f).getroot()

    segments = []
    for i in root:
        id_str = i[0].attrib['href'].split('#')[-1]
        boundary = id_str.split('..')

        if len(boundary) == 2:
            from_w, to_w = boundary
            word_id_pattern = re.compile(r"words(\d+)")

            matched = word_id_pattern.search(from_w)
            if matched is None:
                raise Exception("Couldn't extract word id in segment xml")
            from_w = int(matched.group(1))

            matched = word_id_pattern.search(to_w)
            if matched is None:
                raise Exception("Couldn't extract word id in segment xml")
            to_w = int(matched.group(1))

            segments.append({
                'start': float(i.attrib['transcriber_start']),
                'end': float(i.attrib['transcriber_end']),
                'from_word': from_w,
                'to_word': to_w
            })
        elif len(boundary) == 1:
            # Example: IS1000a.D.words.xml#id(IS1000a.D.words24)
            # simply ignore
            pass
        else:
            raise Exception("Encountered unknown type of word while parsing segments")

    return segments


def parse_words(words_file: Path):
    chk_file(words_file)

    with open(words_file) as f:
        root = ET.parse(f).getroot()

    words: OrderedDict[str, Optional[Dict]] = OrderedDict()

    for i in root:
        ns_prefix = r"{http://nite.sourceforge.net/}"  # handling XML namespace
        w_id = i.attrib[f"{ns_prefix}id"]

        if i.tag != 'w':
            words[w_id] = None
            continue

        # TODO: differentiate punctuation
        if ('starttime' not in i.attrib) or ('endtime' not in i.attrib):
            words[w_id] = None
            continue

        words[w_id] = {
            'start': float(i.attrib['starttime']),
            'end': float(i.attrib['endtime']),
            'content': i.text,
            'punc': True if 'punc' in i.attrib else False
        }
    return words


def join_seg_words(segments: list, words: list):
    result = []
    for i in segments:
        from_idx = i['from_word']
        to_idx = i['to_word']

        tmp = list(filter(lambda x: x is not None, words[from_idx:to_idx + 1]))

        result.append(tmp)
    return result


def make_rttm(meet_id: str, transcripts: Dict[str, dict], speakers: Dict[str, dict]):
    """Transform data into RTTM
    RTTM is the acronym of Rich Transcription Time Marked
    RTTM specification: https://github.com/nryant/dscore/blob/master/README.md#rttm
    """

    # Fields: Type, File ID, Channel ID, Turn Onset, Turn Duration, Orthography Field, Speaker Type, Speaker Name, Confidence Score, Signal Lookahead Time
    # Type should always be SPEAKER
    # Channel ID should always be 1
    # Orthography Field, Speaker Type, Confidence Score, and Signal Lookahead Time should always be <NA>

    result = []

    for spkr_id, data in transcripts[meet_id].items():
        for item in data:
            start = item['start']
            end = item['end']

            result.append((start, ' '.join(
                ['SPEAKER', meet_id, '1', "%.3f" % start, "%.3f" % (end - start), '<NA>', '<NA>',
                 speakers[meet_id][spkr_id]['id'], '<NA>', '<NA>'])))

    result.sort(key=lambda x: x[0])  # sort rttm by start time
    result = list(map(lambda x: x[1], result))  # we only need rttm string

    return result


def run_from_cli():
    argparse = ArgumentParser("ami_transcription_extractor")
    argparse.add_argument("annotation_dir", help="The directory of AMI annotations.", type=Path)
    argparse.add_argument("--output", "-o", default="output", type=Path)
    args = argparse.parse_args()

    # Check whether the directory is valid
    anno_root = args.annotation_dir
    if not anno_root.is_dir():
        raise Exception("Cannot open AMI annotation directory, the input path may be invalid.")

    meeting_xml = anno_root.joinpath('corpusResources', 'meetings.xml')
    if not meeting_xml.is_file():
        raise Exception(
            "Cannot open meeting.xml, either the file is missing or your AMI annotation directory is corrupted.")

    try:
        meeting_et = ET.parse(meeting_xml).getroot()
    except Exception as e:
        raise Exception(f"Error occured while parsing meeting.xml: {e}")

    meetings = [{
        'id': i.attrib['observation'],
        'duration': i.attrib['duration'],
        'type': i.attrib['type']
    } for i in meeting_et]  # A list contains all IDs of meetings with speaker annotation

    speakers: Dict[str, Dict[str, Union[str, int]]] = {}  # A dictionary contains all speakers for every meetings

    # Extract annotated meeting and speakers
    for meet in meeting_et.findall('meeting'):
        m_id = meet.get('observation')
        spkr_info = {}
        for spkr_id in meet.findall('speaker'):
            letter = spkr_id.attrib['nxt_agent']

            spkr_info[letter] = {
                'id': spkr_id.attrib['global_name'],
                'channel': int(spkr_id.attrib['channel'])
            }
        speakers[m_id] = spkr_info

    meet_data = {}

    for meet in meetings:
        meet_id = meet['id']
        tmp = {}

        for spkr_id in speakers[meet_id].keys():
            seg_file = anno_root.joinpath('segments', f"{meet_id}.{spkr_id}.segments.xml")
            words_file = anno_root.joinpath('words', f"{meet_id}.{spkr_id}.words.xml")

            spkr_seg = parse_segments(seg_file)
            spkr_words = list(parse_words(words_file).values())
            tmp[spkr_id] = {
                'segments': spkr_seg,
                'words': spkr_words
            }

        meet_data[meet_id] = tmp


    transcripts = {}
    for meet_id, m_data in meet_data.items():
        tmp = {}
        for spkr, speech in m_data.items():
            grouped_words = join_seg_words(speech['segments'], speech['words'])
            sentences = []

            for i in grouped_words:
                if len(i) == 0:
                    continue
                sentences.append({
                    'start': i[0]['start'],
                    'end': i[-1]['end'],
                    'content': ' '.join(map(lambda x: x['content'], i))
                })

            tmp[spkr] = sentences
        transcripts[meet_id] = tmp

    rttms = {}
    for meet_id, m_data in meet_data.items():
        rttms[meet_id] = '\n'.join(make_rttm(meet_id, transcripts, speakers))

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    for meet_id in rttms:
        with open(output_dir.joinpath(f"{meet_id}.rttm"), mode="w") as f:
            f.write(rttms[meet_id])


if __name__ == "__main__":
    run_from_cli()
