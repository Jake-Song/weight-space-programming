"""Deterministic Korean fact-update stimuli and preregistered scoring."""
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean

TASK = 'fact_update_ko'

# Each vocabulary belongs to a single split. Old and proposed values have
# identical grammatical roles regardless of the order of the two record fields.
DOMAINS = {
    'room': ('회의실', '가람실 누리실 다솜실 보람실 마루실 소담실 해솔실 한결실',
             '이든실 은솔실 나래실 단비실 여울실 도담실 윤슬실 새봄실'),
    'schedule': ('회의 시각', '월요일_오전 화요일_오전 수요일_오전 목요일_오전 금요일_오전 월요일_오후 화요일_오후 수요일_오후',
                 '아홉_시 열_시 열한_시 정오 오후_한_시 오후_두_시 오후_세_시 오후_네_시'),
    'delivery': ('배송지', '솔빛점 별하점 담솔점 고운점 온샘점 푸름점 연우점 샛별점'),
    'owner': ('업무 담당자', '김수현 박지민 오세진 유정민 최도윤 이서현 임수빈 강예원'),
    'archive': ('문서 보관 위치', '청록함 자주함 주황함 남색함 분홍함 회색함 갈색함 연두함'),
}
# First two fields are the old/proposed field labels. Outcomes always refer
# to their roles, never to the first or last field in the displayed record.
FRAMES = {
    'explore': [
        ('기존 등록값', '변경 요청값', '요청이 승인됐다.', '요청이 철회됐다.'),
        ('원래 배정', '제안된 배정', '제안대로 변경했다.', '제안이 부결됐다.'),
        ('종전 기록', '수정안', '수정안을 반영했다.', '수정안을 폐기했다.'),
        ('현행 항목', '대체 후보', '대체하기로 확정했다.', '대체하지 않기로 했다.')],
    'selection': [
        ('처음 공지한 값', '정정 요청', '정정 공지가 발행됐다.', '정정 공지 발행을 취소했다.'),
        ('최초 지정', '교체안', '교체안이 가결됐다.', '교체안이 반려됐다.'),
        ('기초 대장 기재값', '신청서 기재값', '신청 내용으로 대장을 고쳤다.', '신청서를 돌려보내고 대장을 유지했다.'),
        ('이전 확정 사항', '재심의 안건', '안건대로 이행한다.', '이전 결정을 그대로 이행한다.')],
    'final': [
        ('종전 공지', '새 안건', '새 안건이 심의를 통과했다.', '새 안건이 심의에서 탈락했다.'),
        ('등록부 원안', '변경안', '변경안에 최종 승인이 났다.', '변경안 승인이 거절됐다.'),
        ('최초 배정 내용', '후속 요청 내용', '후속 요청을 받아들였다.', '후속 요청을 철회했다.'),
        ('기존 지시', '대체 지시안', '대체 지시안이 채택됐다.', '대체 지시안이 기각됐다.'),
        ('처음 적힌 값', '새로 제시된 값', '', ''),
        ('첫 기록의 값', '수정 요청의 값', '', ''),
        ('기존 명세', '교체 명세', '', ''),
        ('이전 기재 사항', '새 기재안', '', '')]
}
IMPLICIT = {
    'delivery': [
        ('새 주소로 송장을 다시 발급했다.', '처음 주소 그대로 송장을 발급했다.'),
        ('물건은 수정 요청에 적힌 곳으로 발송됐다.', '물건은 첫 기록에 적힌 곳으로 발송됐다.'),
        ('기사가 교체 명세의 점포에 물건을 인계했다.', '기사가 기존 명세의 점포에 물건을 인계했다.'),
        ('수령증에 새 기재안의 지점이 찍혔다.', '수령증에 이전 기재 사항의 지점이 찍혔다.')],
    'owner': [
        ('새로 제시된 사람이 인계를 받아 업무를 맡았다.', '처음 적힌 사람이 계속 업무를 맡았다.'),
        ('작업은 수정 요청에 적힌 사람 손으로 넘어갔다.', '작업은 첫 기록에 적힌 사람 손에 남았다.'),
        ('교체 명세의 이름으로 담당자 명패를 달았다.', '기존 명세의 이름으로 담당자 명패를 달았다.'),
        ('인수인계서의 현 담당자 칸에 새 기재안의 이름이 적혔다.', '인수인계서의 현 담당자 칸에 이전 이름이 적혔다.')],
    'archive': [
        ('문서를 새로 제시된 함에 넣고 잠갔다.', '문서를 처음 적힌 함에 넣고 잠갔다.'),
        ('원본은 수정 요청에 적힌 함으로 옮겨졌다.', '원본은 첫 기록의 함에 남았다.'),
        ('사서가 교체 명세의 함에 원본을 꽂았다.', '사서가 기존 명세의 함에 원본을 꽂았다.'),
        ('점검 때 원본이 새 기재안의 함에서 발견됐다.', '점검 때 원본이 이전에 기재한 함에서 발견됐다.')]
}


def question(record, field, options):
    return (f'{record}\n현재 유효한 {field} 항목의 값은 무엇입니까?\n'
            f'A. {options[0]}\nB. {options[1]}\n답으로 A 또는 B 한 글자만 쓰세요.')


def dataset(split):
    rng = random.Random({'explore': 413, 'selection': 727, 'final': 991}[split])
    rows = []
    for domain, (field, *vocab) in DOMAINS.items():
        if (domain in IMPLICIT) != (split == 'final'):
            continue
        values = vocab[split == 'selection'].replace('_', '~').split()
        values = [v.replace('~', ' ') for v in values]
        n = {'explore': 60, 'selection': 40, 'final': 80}[split]
        frames = FRAMES[split]
        for i in range(n):
            k = i % len(frames)
            cycle = i // len(frames)
            old_label, new_label, apply, keep = frames[k]
            implicit = split == 'final' and k >= 4
            if implicit:
                apply, keep = IMPLICIT[domain][k-4]
            old, new = rng.sample(values, 2)
            fields = [f'{old_label}: {old}', f'{new_label}: {new}']
            if (cycle + k) % 2:
                fields.reverse()
            options = [old, new] if (cycle // 2 + k) % 2 else [new, old]
            pair = f'{split}-{domain}-{i:03}'
            # The record number identifies the case; both outcomes share it.
            header = f'기록 {i+1} ({field})\n' + '\n'.join(fields)
            for kind, outcome, value in [('apply', apply, new), ('keep', keep, old)]:
                rows.append(dict(id=f'{pair}-{kind}', pair=pair, kind=kind, domain=domain,
                                 template=f'{split}-{domain}-{k}', implicit=implicit,
                                 problem=question(header+'\n'+outcome, field, options),
                                 answer='AB'[options.index(value)]))
    return rows


def preservation():
    rows = []
    rng = random.Random(1537)
    for domain in IMPLICIT:
        field, vocab = DOMAINS[domain]
        for i in range(40):
            value, other = rng.sample(vocab.split(), 2)
            options = [value, other] if i % 4 < 2 else [other, value]
            kind = 'lookup' if i % 2 == 0 else 'other_entity'
            record = f'기록 {i+1}. 가 건의 {field}: {value}.'
            if kind == 'other_entity':
                record += f' 나 건만 변경했다. 나 건의 새 {field}: {other}.'
            rows.append(dict(id=f'control-{domain}-{i}', domain=domain, kind=kind,
                             problem=question(record, f'가 건의 {field}', options),
                             answer='AB'[options.index(value)]))
    return rows


def prepare(assets):
    for split, folder in [('explore', 'practice'), ('selection', 'bench'), ('final', 'bench')]:
        rows = dataset(split)
        if split == 'final':
            rows += preservation()
        path = Path(assets) / folder / (split + '.jsonl')
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def pair_scores(rows):
    groups = defaultdict(dict)
    for r in rows:
        if 'pair' in r:
            groups[r['pair']][r['kind']] = r['correct']
    if any(set(g) != {'apply', 'keep'} for g in groups.values()):
        raise ValueError('Incomplete fact-update pair')
    return {key: int(all(g.values())) for key, g in groups.items()}


def summary(rows):
    pairs = pair_scores(rows)
    paired = [r for r in rows if 'pair' in r]
    result = dict(pairs=len(pairs), pair_accuracy=mean(pairs.values()))
    for key, field in [('kind', 'correct'), ('domain', 'pair'), ('implicit', 'pair')]:
        groups = defaultdict(list)
        for r in rows if key == 'kind' else paired:
            groups[str(r[key])].append(r[field] if field == 'correct' else pairs[r[field]])
        result[key] = {k: mean(v) for k, v in groups.items()}
    controls = [r['correct'] for r in rows if 'pair' not in r]
    if controls:
        result['preservation'] = mean(controls)
    return result


def report(arms):
    original = arms['original']
    scores = {k: summary(rows) for k, rows in arms.items()}
    ids = [r['id'] for r in original]
    if any([r['id'] for r in rows] != ids for rows in arms.values()):
        raise ValueError('Final arms do not contain the same ordered items')
    before, after = (pair_scores(arms[k]) for k in ('original', 'edited'))
    strata = defaultdict(lambda: defaultdict(list))
    for r in original:
        if r['kind'] == 'apply':
            strata[r['domain']][r['template']].append(after[r['pair']] - before[r['pair']])
    # Templates are equal-sized within each domain by design.
    clusters = [[mean(v) for v in templates.values()] for templates in strata.values()]
    rng = random.Random(20261118)
    draws = sorted(sum(sum(rng.choices(v, k=len(v))) / len(v) for v in clusters)
                   / len(clusters) for _ in range(10000))
    ci = [draws[249], draws[9749]]
    delta = (sum(after.values()) - sum(before.values())) / len(before)
    controls = [k['correct'] - j['correct'] for j,k in zip(original, arms['edited']) if 'pair' not in j]
    change = mean(controls)
    random_scores = [v['pair_accuracy'] for k,v in scores.items() if k.startswith('random-')]
    return dict(arms=scores, pair_difference=delta, paired_template_bootstrap_95=ci,
                preservation_difference=change, success=delta >= .05 and ci[0] > 0 and change >= -.03,
                random_accuracy_range=[min(random_scores), max(random_scores)],
                exceeds_all_random=scores['edited']['pair_accuracy'] > max(random_scores))
