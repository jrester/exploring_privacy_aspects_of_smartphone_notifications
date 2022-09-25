import csv
from typing import Dict
import statistics
from collections import Counter
from enum import Enum
from tabulate import tabulate
from scipy import stats
import datetime

CUR_YEAR = datetime.datetime.now().year

ALPHA = 5e-3

class Tag(Enum):
    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.value < other.value


class Checkbox(Tag):
    UNECHECKED = 1
    CHECKED = 2


class Gender(Tag):
    FEMALE = 1
    MALE = 2
    DIVERS = 3
    CUSTOM = 4
    NA = 5
    SKIPPED = -9


class Nation(Tag):
    GERMAN = "B208_12"
    TURKISH = "B208_01"
    EAST_EUROPEAN = "B208_05"
    OTHER_EUROPEAN = "B208_06"
    AFRICAN = "B208_09"
    AMERICAN = "B208_07"
    ASIAN = "B208_08"
    CUSTOM = "B208_10"
    NA = "B208_11"


class Education(Tag):
    NONE = 1
    SCHUELER = 9
    HAUPTSCHULE = 3
    REALSCHULE = 4
    LEHRE = 5
    FACHABITUR = 6
    ABITUR = 7
    FACHHOCHSCHULE = 8
    DOKTOR = 11
    OTHER = 10
    NA = 12
    SKIPPED = -9

class Employment(Tag):
    FULLTIME = 1
    PARTTIME = 2
    SELF_EMPLOYED = 3
    SEARCHING_FOR_WORK = 4
    NOT_SEARCHING = 5
    BACHELOR = 6
    MASTER = 7
    PENSION = 8
    DISABLED = 9
    OTHER = 10
    NA = 11
    SKIPPED = -9
    AUSBILDUNG = 12


class OS(Tag):
    ANDROID = 1
    IOS = 2
    OTHER = 3
    SKIPPED = -9


class ScalePrivacyConcernedApp(Tag):
    COMPLETELY_UNCONSERNED = 1
    SOMEWHAT_UNCONSERNED = 2
    NEUTRAL = 3
    SOMEWHAT_CONSERNED = 4
    EXTREMELY_CONSERNED = 5
    NO_APP_IN_CATEGORY = -1
    NA = -9


class ScalePrivacyConcern(Tag):
    COMPLETELY_UNCONSERNED = 1
    SOMEWHAT_UNCONSERNED = 5
    NEUTRAL = 4
    SOMEWHAT_CONSERNED = 3
    EXTREMELY_CONSERNED = 2
    NA = 6
    NOTIFCATION_NOT_FROM_ANOTHER_PERSON = 7
    SKIPPED = -9


class AppCategory(Tag):
    INSTANT_MESSAGING = 1
    SOCIAL_MEDIA = 2
    CALENDAR = 3
    E_MAIL = 4
    BANKING = 5
    HEALTH = 6
    DATING = 7


class SharingScale(Tag):
    NEVER = 1
    LESS_THAN_ONCE_A_WEEK = 2
    MULTIPLE_TIMES_A_WEEK = 3
    ONCE_PER_DAY = 4
    MULTIPLE_TIMES_A_DAY = 5
    NOT_APPLICABLE = -1
    NA = -9


class Person(Tag):
    SPOUSE = 1
    SILBLINGS = 2
    PARENTS = 3
    CHILD = 4
    FRIENDS = 5
    COLLEAGUE = 6
    STRANGER = 7


class ControllMechanism(Tag):
    DND = 1
    VIBRATION = 2
    OFF = 3
    AIRPLANE = 4
    NOTIFICATION_APPS = 5


class Places(Tag):
    HOME = 1
    MEETINGS = 2
    CLASSES = 3
    SEMINARS = 4
    CAFETERIA_OR_RESTAURANT = 5
    SOCIAL = 6
    PUBLIC_TRANSPORT = 7
    OTHER = 8


class FrequencyWeek(Tag):
    NEVER = 1
    SOMETIMES_PER_WEEK = 2
    ONCE_A_DAY = 3
    MULTIPLE_TIMES_A_DAY = 4
    NA = -1
    SKIPPED = -9


class Frequency(Tag):
    NEVER = 1
    RARE = 2
    SOMETIMES = 3
    ALWAYS = 4
    NA = -1
    SKIPPED = -9


class SharingPrequesits(Tag):
    CLOSE_APPS = "A404_03"
    DELETE_ALL_NOTIFICATIONS = "A404_04"
    GUEST_MODE = "A404_05"
    APP_LOCK = "A404_06"
    OBSERVE = "A404_07"
    CONTROL_APPS = "A404_08"
    OTHER = "A404_09"
    NOT_SHARING = "A404_10"
    NOTHING = "A404_11"


class YesNo(Tag):
    YES = 1
    NO = 2
    OTHER = 3
    NA = 4
    SKIPPED = -9


class BadExperienceReason(Tag):
    HANDED_OUT_PHONE = 1
    PHONE_NOT_LOCKED = 3
    NO_MEASURMENTS = 4
    PRIVATE_INFORMATION = 7
    SHARING_PRESENTATION = 5
    OTHER = 8
    NA = 9


ColMappingBadExperienceReason = {f"A603_0{e.value}": e for e in BadExperienceReason}

ColMappingClearNotifications = {f"A405_0{p.value}": p for p in Person}

ColMappingSharingPrequesits = {s.value: s for s in SharingPrequesits}

ColMappingPlaces = {f"A401_0{p.value}": p for p in Places}

ColMappingControllMechanism = {f"A304_0{m.value}": m for m in ControllMechanism}

ColMappingSharing = {f"A402_0{person.value}": person for person in Person}

ColMappingFamilyConcern = {f"A701_0{app.value}": app for app in AppCategory}

ColMappingColleagueConcern = {f"A703_0{app.value}": app for app in AppCategory}

BadExperiences = {
    1: "0",
    2: "1-5",
    3: "1-5",
    4: "6-10",
    5: "11-20",
    6: ">20",
    7: "NA",
    -9: "SKIPPED",
}

# List of participants that gave thoughtfull answers
QUALITITVE_ANSWERS = [188, 199, 202, 280, 296, 309, 327, 331, 348, 350, 366, 399, 434, 454, 289, 298, 316, 318, 349, 370, 398, 400, 428, 468]


def sort_dict(d):
    return dict(sorted(d.items(), key=lambda item: item[0]))


class Header:
    def __init__(self, header):
        self._header = header
        self._col_mapping: Dict[str, int] = dict()

    def get_col(self, key):
        if key in self._col_mapping:
            return self._col_mapping[key]
        else:
            for i in range(len(self._header)):
                if self._header[i] == key:
                    self._col_mapping[key] = i
                    return i
            raise RuntimeError("Invalid column: {}".format(key))


class Row:
    # Row allows transparent access to columns by name
    def __init__(self, header: Header, row):
        self._header = header
        self._row = row

    def __getitem__(self, key):
        return self._row[self._header.get_col(key)]


def first_attention_check(row):
    return len(row["A704"]) > 0 and int(row["A704"]) == 2


def second_attention_check(row):
    return len(row["A901_05"]) > 0 and int(row["A901_05"]) == 5


def is_valid(row, qualtitve_check = True):
    valid_first = first_attention_check(row)
    valid_second = second_attention_check(row)
    valid = valid_first and valid_second
    case = int(row['\ufeffCASE'])
    if qualtitve_check:
        if case in QUALITITVE_ANSWERS:
            if not valid:
                failed_text = "both" if not valid_first and not valid_second else ("first" if not valid_first else "second") 
                print(f"Participant {case}: invalid attention checks, but gave thoughtfull answers, therefore including in dataset. Failed {failed_text} attention check.")
                return True

    return valid
        
class Dataset:
    def __init__(self, file_path):
        self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, encoding="utf-16-le") as f:
            rows = list(csv.reader(f, delimiter=";", quotechar='"'))
            self.header = Header(rows[0])
            self.rows = [Row(self.header, r) for r in rows[1:]]

    def process(self):
        self.total_count = len(self.rows)
        self.count_by_source_before_filter = self.per_source()
        self.filter_valid()
        self.valid_count = len(self.rows)
        self.count_by_source_after_filter = self.per_source()
        
    def drop_out_rates_by_source(self):
        out = {}
        for key, value in self.count_by_source_after_filter.items():
            out[key] = 1- (value/self.count_by_source_before_filter[key])

        return out

    def filter_valid(self):
        # Filter without including qualitive answers
        rows_cnt = len(self.rows)
        rows = list(filter(lambda r: is_valid(r, False), self.rows))
        print(f"Without qualitive inclusion: n = {len(rows)}, p = {len(rows)/rows_cnt}")

        self.rows = list(filter(is_valid, self.rows))

    def _cnt_col(self, col, tranformer=lambda v: v, filter=lambda _: True):
        cnt = Counter()
        for row in self.rows:
            if len(row[col]) > 0:
                try:
                    key = tranformer(row[col])
                except ValueError as e:
                    raise ValueError(
                        f"Transformer failed for column {col} with value {row[col]}"
                    )
                if filter(key):
                    cnt[key] += 1
        return cnt

    def _cnt_to_perc(self, cnt: Counter, total: int = None):
        """calculate the percentages for all keys in a Counter object"""
        out = {}
        for key in cnt.keys():
            if total is None:
                total = cnt.total()
            out[key] = round(cnt[key] / total * 100, 1)
        return out

    def _analyze_col(self, col: str, transformer=lambda v: v, filter=lambda _: True):
        cnt = self._clean_cnt(self._cnt_col(col, transformer, filter))
        perc = self._cnt_to_perc(cnt)
        return (sort_dict(cnt), perc)

    def _clean_cnt(self, cnt: Counter):
        if cnt[""] > 0:
            cnt.pop("")
        return cnt

    def per_source(self):
        return self._cnt_col("REF")

    def _age(self):
        age_list = []
        for row in self.rows:
            if len(row["B204_01"]) > 0:
                age_list.append(int(row["B204_01"]))
        return age_list

    def max_birth_year(self):
        return max(self._age())

    def min_birth_year(self):
        return min(self._age())

    def mean_age(self):
        return round(statistics.mean(self._age()), 2)

    def median_age(self):
        return round(statistics.median(self._age()), 2)

    def gender_total(self):
        return self._cnt_col("B201", lambda v: Gender(int(v)))

    def gender_perc(self):
        return self._cnt_to_perc(self.gender_total())

    def multiple_nations(self):
        cnt = Counter()
        for row in self.rows:
            num_nations = 0
            for nation in Nation:
                # 2 is the code for checkbox ticked
                if row[nation.value] == "2":
                    num_nations += 1
            cnt[num_nations] += 1
        return cnt, self._cnt_to_perc(cnt)

    def nation(self):
        out = {}
        for nation in Nation:
            cnt = self._clean_cnt(self._cnt_col(nation.value))
            perc = self._cnt_to_perc(cnt)
            if "2" in cnt.keys():
                # 2 is the code for checkbox ticked
                out[nation] = (perc["2"], cnt["2"])
        return out

    def education(self):
        return self._analyze_col("B211", lambda v: Education(int(v)))

    def number_of_negative_exp(self):
        return self._analyze_col("A601", lambda v: BadExperiences[int(v)])

    def number_of_negative_exp_over_zero(self):
        res = self._analyze_col("A601", lambda v: int(v))
        filter_fn = lambda d: {k: v for k, v in d.items() if k > 1 and k < 7}
        return (filter_fn(res[0]), filter_fn(res[1]))

    def _analyse_multi_col(self, col_mapping: dict, transformer, filter=lambda _: True):
        out = {}
        for col, key in col_mapping.items():
            cnt, perc = self._analyze_col(col, transformer, filter)
            out[key] = (cnt, perc)
        return out

    def concern_family(self):
        return self._analyse_multi_col(
            ColMappingFamilyConcern,
            lambda v: ScalePrivacyConcernedApp(int(v)),
            lambda v: v != ScalePrivacyConcernedApp.NA,
        )

    def concern_colleague(self):
        return self._analyse_multi_col(
            ColMappingColleagueConcern, lambda v: ScalePrivacyConcernedApp(int(v))
        )

    def os(self):
        return self._analyze_col("OS", lambda v: OS(int(v)))

    def privacy_concern_for_self(self):
        return self._analyze_col("A507", lambda v: ScalePrivacyConcern(int(v)))

    def privacy_concern_for_other(self):
        return self._analyze_col("A510", lambda v: ScalePrivacyConcern(int(v)), filter=lambda v: v != ScalePrivacyConcern.NOTIFCATION_NOT_FROM_ANOTHER_PERSON)

    def _filter_app_concern_map(self, out):
        for app in out.keys():
            if ScalePrivacyConcernedApp.NO_APP_IN_CATEGORY in out[app][0]:
                del out[app][0][ScalePrivacyConcernedApp.NO_APP_IN_CATEGORY]
            if ScalePrivacyConcernedApp.NA in out[app][0]:
                del out[app][0][ScalePrivacyConcernedApp.NA]

    def _calc_mean_app_concern_map(self, concern_app_map):
        _map = {
            app: [code.value for code, n in cnt.items() for _ in range(0, n)]
            for app, (cnt, _) in concern_app_map.items()
        }
        _map = {
            app: (round(statistics.mean(l), 2), round(statistics.stdev(l), 2))
            for app, l in _map.items()
        }
        return _map

    def mean_privacy_concern_by_app_category_for_family_member(self):
        concern_app_map = self.concern_family()
        self._filter_app_concern_map(concern_app_map)
        return self._calc_mean_app_concern_map(concern_app_map)

    def mean_privacy_concern_by_app_category_for_colleague(self):
        concern_app_map = self.concern_colleague()
        self._filter_app_concern_map(concern_app_map)
        return self._calc_mean_app_concern_map(concern_app_map)

    def frequency_device_sharing(self):
        return self._analyse_multi_col(
            ColMappingSharing, lambda v: SharingScale(int(v)),
            filter=lambda v: v!=SharingScale.NOT_APPLICABLE and v!=SharingScale.NA
        )

    def _device_meachanism_transformer(self, value):
        value = int(value)
        if value == -9:
            return ""
        elif value == 1:
            return ""
        else:
            return value - 1

    def device_mechanisms(self):
        res = self._analyse_multi_col(
            ColMappingControllMechanism, self._device_meachanism_transformer
        )
        out = dict()
        for mechanism, (cnt, _) in res.items():
            out[mechanism] = round(
                statistics.mean([x for x, n in cnt.items() for _ in range(0, n)]), 2
            )

        return out

    def device_mechanism_correlated_to_negative_experiences(self):
        mechanism_to_neg_map = {m: {} for m in ControllMechanism}
        for row in self.rows:
            neg_exp_group = int(row["A601"])
            if neg_exp_group == 5:
                continue

            for col, m in ColMappingControllMechanism.items():
                m_employed_hours = self._device_meachanism_transformer(row[col])

                if m_employed_hours != "":

                    if neg_exp_group not in mechanism_to_neg_map[m]:
                        mechanism_to_neg_map[m][neg_exp_group] = []

                    mechanism_to_neg_map[m][neg_exp_group].append(m_employed_hours)

        res = {}
        for m in ControllMechanism:
            res[m] = stats.kruskal(*mechanism_to_neg_map[m].values())

        return res

    def times_another_person_can_read_notifications_by_place(self):
        return self._analyse_multi_col(
            ColMappingPlaces,
            lambda v: FrequencyWeek(int(v)),
            lambda v: v != FrequencyWeek.NA and v != FrequencyWeek.SKIPPED,
        )

    def _expand_counter_to_vals_to_list(self, cnt):
        return [scale.value for scale, n in cnt.items() for _ in range(0, n)]

    def _truncate_to_same_length(self, l1, l2):
        length = min(len(l1), len(l2))
        return (l1[:length], l2[:length])

    def privacy_concern_wilcox(self):
        _out = dict()

        def _filter(cnt):
            if ScalePrivacyConcernedApp.NO_APP_IN_CATEGORY in cnt:
                del cnt[ScalePrivacyConcernedApp.NO_APP_IN_CATEGORY]
            if ScalePrivacyConcernedApp.NA in cnt:
                del cnt[ScalePrivacyConcernedApp.NA]
            
        for family_col, app in ColMappingFamilyConcern.items():
            colleague_col = [
                col for col, _app in ColMappingColleagueConcern.items() if _app == app
            ][0]

            family_cnt = self._cnt_col(
                family_col, lambda v: ScalePrivacyConcernedApp(int(v))
            )
            colleague_cnt = self._cnt_col(
                colleague_col, lambda v: ScalePrivacyConcernedApp(int(v))
            )

            _filter(family_cnt)
            _filter(colleague_cnt)

            family_l = self._expand_counter_to_vals_to_list(family_cnt)
            colleague_l = self._expand_counter_to_vals_to_list(colleague_cnt)

            # might truncation lead to lost data
            family_l, colleague_l = self._truncate_to_same_length(family_l, colleague_l)

            _out[app] = stats.wilcoxon(family_l, colleague_l, correction=True, zero_method="pratt")

        return _out

    def _perceived_survaillance(self, row):
        _vals = []
        for i in range(1, 4):
            v = int(row[f"A901_0{i}"])
            if v == -9:
                return None
            _vals.append(v)

        return statistics.mean(_vals)

    def _perceived_intrusion(self, row):
        _vals = [int(row["A901_04"]), int(row["A901_06"]), int(row["A901_07"])]
        for i in _vals:
            if i == -9:
                return None

        return statistics.mean(_vals)

    def _dev_mechanism_mapping(self, row):
        _map = {}
        for col, m in ColMappingControllMechanism.items():
            v = self._device_meachanism_transformer(row[col])
            if isinstance(v, int):
                _map[m] = v

        return _map

    def _neg_exp_group(self, row):
        return BadExperiences[int(row["A601"])]

    def _raw_muipc_neg_exp_and_dev_ctrl_mechanism_mapping(self):
        res = {}
        for row in self.rows:
            surv_score = self._perceived_survaillance(row)
            intr_score = self._perceived_intrusion(row)
            neg_exp_group = self._neg_exp_group(row)
            dev_mech_map = self._dev_mechanism_mapping(row)

            if neg_exp_group not in res:
                res[neg_exp_group] = dict()

            if surv_score is not None:
                if "surv_score" not in res[neg_exp_group]:
                    res[neg_exp_group]["surv_score"] = []

                res[neg_exp_group]["surv_score"].append(surv_score)

            if intr_score is not None:
                if "intr_score" not in res[neg_exp_group]:
                    res[neg_exp_group]["intr_score"] = []

                res[neg_exp_group]["intr_score"].append(intr_score)

            if "dev_mech_map" not in res[neg_exp_group]:
                res[neg_exp_group]["dev_mech_map"] = dict()

            for m, hours in dev_mech_map.items():
                if m not in res[neg_exp_group]["dev_mech_map"]:
                    res[neg_exp_group]["dev_mech_map"][m] = list()
                res[neg_exp_group]["dev_mech_map"][m].append(hours)

            if "n" not in res[neg_exp_group]:
                res[neg_exp_group]["n"] = 0

            res[neg_exp_group]["n"] += 1

        return res

    def spearman_correlation_intr_neg_exp(self):
        l = list()
        s = []
        n = []
        for row in self.rows:
            intr = self._perceived_intrusion(row)
            neg_exp_group = int(row["A601"])

            if neg_exp_group >= 0 and neg_exp_group < 7:
                l.append([intr, neg_exp_group])
                s.append(intr)
                n.append(neg_exp_group)

        return stats.spearmanr(stats.rankdata(l).reshape(len(l), 2))

    def spearmen_correlation_surv_neg_exp(self):
        l = list()
        s = []
        n = []
        for row in self.rows:
            surv = self._perceived_survaillance(row)
            neg_exp_group = int(row["A601"])
            if neg_exp_group >= 0 and neg_exp_group < 7:
                if surv is not None:
                    l.append([surv, neg_exp_group])
                    s.append(surv)
                    n.append(neg_exp_group)
        
        return stats.spearmanr(stats.rankdata(l).reshape(len(l), 2))


    def muipc_neg_exp_and_dev_ctrl_mechanism_mapping(self):
        raw = self._raw_muipc_neg_exp_and_dev_ctrl_mechanism_mapping()

        for neg_exp_group, data in raw.items():
            raw[neg_exp_group]["intr_score"] = statistics.mean(data["intr_score"])
            raw[neg_exp_group]["surv_score"] = statistics.mean(data["surv_score"])

            for m, hours in data["dev_mech_map"].items():
                raw[neg_exp_group]["dev_mech_map"][m] = statistics.mean(hours)

        return sort_dict(raw)

    def _digital_difficulty_score(self, row):
        _vals = []
        for i in range(1, 6):
            v = int(row[f"B102_0{i}"])
            if v == -9:
                return None

            _vals.append(v)

        return statistics.mean(_vals)

    def _raw_neg_exp_dds(self):
        res = {}
        for row in self.rows:
            neg_exp_group = self._neg_exp_group(row)
            dds = self._digital_difficulty_score(row)
            if dds is not None:
                if neg_exp_group not in res:
                    res[neg_exp_group] = []

                res[neg_exp_group].append(dds)

        return sort_dict(res)

    def neg_exp_dds(self):
        raw = self._raw_neg_exp_dds()

        res = dict()
        for neg_exp_group, dds_list in raw.items():
                if neg_exp_group not in res:
                    res[neg_exp_group] = dict()
                    res[neg_exp_group]["n"] = len(dds_list)

                res[neg_exp_group]["mean"] = round(statistics.mean(dds_list), 2)
                res[neg_exp_group]["median"] = round(statistics.median(dds_list), 2)

                if len(dds_list) > 1:
                    res[neg_exp_group]["sd"] = round(statistics.stdev(dds_list), 2)
                else:
                    res[neg_exp_group]["sd"] = None

        return res

    def _neg_exp_by_os(self):
        _mapping = {OS.ANDROID: [], OS.IOS: []}
        for row in self.rows:
            os = OS(int(row["OS"]))
            if os == OS.ANDROID or os == OS.IOS:
                neq_exp = int(row["A601"])
                if neq_exp >= 0 and neq_exp < 7:
                    _mapping[os].append(neq_exp)

        return _mapping

    def correlation_os_neg_exp(self):
        _mapping = self._neg_exp_by_os()
        return stats.mannwhitneyu(
            _mapping[OS.IOS], _mapping[OS.ANDROID], method="exact"
        )

    def correlation_age_net_exp(self):
        arr = []
        for row in self.rows:
            neq_exp = int(row["A601"])
            if neq_exp >= 0 and neq_exp < 7:
                age = row["B204_01"]
                if len(age) > 0:
                    arr.append((int(age), neq_exp))

        arr.sort(key=lambda t: t[0], reverse=True)

        age_axis = [t[0] for t in arr]
        y_axis = [t[1] for t in arr]

        return stats.pearsonr(age_axis, y_axis)

    def correlation_neg_exp_dds(self):
        n = []
        d = []
        for row in self.rows:
            neg_exp = int(row["A601"])
            dds = self._digital_difficulty_score(row)
            if neg_exp >= 0 and neg_exp < 7 and neg_exp != 5:
                n.append(neg_exp)
                d.append(dds)

        return stats.pearsonr(d, n)

    def device_mechanisms_used(self):
        a = self._analyse_multi_col(
            ColMappingControllMechanism, self._device_meachanism_transformer
        )
        cnt = Counter()
        for m in ControllMechanism:
            cnt[m] = sum(a[m][0].values())
        return sort_dict(cnt), self._cnt_to_perc(cnt, total=self.valid_count)
        
CHAPTER = 1

def print_header(title, i=20):
    global CHAPTER
    print("\n")
    print("===" * i, CHAPTER, title, "===" * i)
    print("\n")
    CHAPTER += 1


def print_analysis(title, analysis):
    print_header(title)
    total, perc = analysis
    table = [[key, val, f"{perc[key]}%"] for key, val in total.items()]
    print(tabulate(table))


def print_multi_col_analysis(title, analysis):
    print_header(title)

    rows = {"key": []}
    for _, (cnt, _) in analysis.items():
        for key in cnt.keys():
            if rows.get(key) is None:
                rows[key] = []

    fstr = "{:<2} {:<3}%"
    for key, (cnt, perc) in analysis.items():
        rows["key"].append(key)
        for k in cnt.keys():
            rows[k].append(fstr.format(cnt[k], perc[k]))

    print(tabulate(rows, headers="keys"))


def print_age(data_set):
    mean = data_set.mean_age()
    median = data_set.median_age()
    print(f"Mean birth year: {mean} ({CUR_YEAR-mean})")
    print(f"Median birth year: {median} ({CUR_YEAR-median})")
    print(
        f"Max birth year: {data_set.max_birth_year()} ({CUR_YEAR-data_set.max_birth_year()}), Min birth year: {data_set.min_birth_year()} ({CUR_YEAR-data_set.min_birth_year()})"
    )


def print_gender_distribution(data_set):
    gender_distribution_total = data_set.gender_total()
    gender_distribution_perc = data_set.gender_perc()
    for key, val in gender_distribution_total.items():
        print(f"{key}: {val} ({round(gender_distribution_perc[key], 2)}%)")


def print_nation(data_set):
    nation = data_set.nation()
    table = [[n, val[1], f"{val[0]}%"] for n, val in nation.items()]
    print(tabulate(table))

    print_analysis("Multiple Nations", data_set.multiple_nations())


def print_education(data_set):
    print_analysis("Education", data_set.education())

def print_employment(data_set: Dataset):
    print_analysis("Employment", data_set._analyze_col("B21_3", lambda v: Employment(int(v))))

def print_number_of_negative_exp(data_set: Dataset):
    neg_exp_analysis = data_set.number_of_negative_exp()
    print_analysis("Number of Negative Experiences", neg_exp_analysis)
    neg_exp_over_one = data_set.number_of_negative_exp_over_zero()
    print(
        f"Over zero: n = {sum(neg_exp_over_one[0].values())}, p = {sum(neg_exp_over_one[1].values())}"
    )
    print_header("Correlation between OS and negative experiences")
    print(data_set.correlation_os_neg_exp())
    print_header("Correlation between Age and negative experiences")
    print(data_set.correlation_age_net_exp())


def print_os(data_set):
    print_analysis("OS", data_set.os())


def print_concern_family(data_set):
    print_multi_col_analysis("Concern Family", data_set.concern_family())


def print_concern_colleague(data_set):
    print_multi_col_analysis("Concern Colleague", data_set.concern_colleague())


def print_concern_self(data_set):
    print_analysis(
        "Last notification information disclosure concern for self",
        data_set.privacy_concern_for_self(),
    )


def print_concern_other(data_set):
    print_analysis(
        "Last notification information diclosure concern for other",
        data_set.privacy_concern_for_other(),
    )


def print_mean_concern_family(data_set):
    print_header("Mean concern for family member")
    res = data_set.mean_privacy_concern_by_app_category_for_family_member()
    table = [[app, mean, sd] for app, (mean, sd) in res.items()]
    print(tabulate(table, headers=["App", "mean", "sd"]))


def print_mean_conern_colleage(data_set):
    print_header("Mean concern for colleague")
    res = data_set.mean_privacy_concern_by_app_category_for_colleague()
    table = [[app, mean, sd] for app, (mean, sd) in res.items()]
    print(tabulate(table, headers=["App", "mean", "sd"]))


def print_device_sharing(data_set):
    print_multi_col_analysis("Device Sharing", data_set.frequency_device_sharing())


def print_device_mechanism(data_set):
    print_header("Device Mechanisms")
    res = data_set.device_mechanisms()
    table = [[key, val] for key, val in res.items()]
    print(tabulate(table, headers=["Mechanism", "hours/day"]))


def print_device_mechanism_correlated_to_neq_exp(data_set):
    print_header("Correlation Between Negative Experiences and Device Mechanism")
    (
        kruskal,
        mean_values,
    ) = data_set.device_mechanism_correlated_to_negative_experiences()
    table = [[key, val.pvalue, val.statistic] for key, val in kruskal.items()]
    print(tabulate(table, headers=["Mechanism", "pvalue", "statistic"]))


def print_privacy_concern_wilcoxon(data_set: Dataset):
    print_header("Privacy Correlation with wilcoxon between Family and Colleagues")
    res = data_set.privacy_concern_wilcox()
    table = [[app, round(val.pvalue * 100, 2), val.statistic] for app, val in res.items()]
    print(tabulate(table, headers=["App Category", "pvalue %", "statistic"]))


def print_neg_exp_surv_intr_dev_mech(data_set: Dataset):
    print_header(
        "Grouping by Number of negativ experiences, Intrusion Score, Survaillance Score, Device Control Mechanisms and Number of answers",
        i=5,
    )
    res = data_set.muipc_neg_exp_and_dev_ctrl_mechanism_mapping()

    mec = lambda v, m: v["dev_mech_map"].get(m)

    table = [
        [
            k,
            v["n"],
            v["surv_score"],
            v["intr_score"],
            mec(v, ControllMechanism.DND),
            mec(v, ControllMechanism.VIBRATION),
            mec(v, ControllMechanism.OFF),
            mec(v, ControllMechanism.AIRPLANE),
            mec(v, ControllMechanism.NOTIFICATION_APPS),
        ]
        for k, v in res.items()
    ]

    print(
        tabulate(
            table,
            headers=[
                "Number Neg Exp",
                "n",
                "SURV",
                "INTR",
                "DND",
                "VIBRATION",
                "OFF",
                "AIRPLANE",
                "Notification Apps",
            ],
        )
    )
    print("Intr", data_set.spearman_correlation_intr_neg_exp())
    print("Surv", data_set.spearmen_correlation_surv_neg_exp())


def print_neg_exp_dds(data_set: Dataset):
    print_header("Digital Difficulty and Negativ experiences")
    res = data_set.neg_exp_dds()
    table = [[k, v["mean"], v["median"], v["sd"], v["n"]] for k, v in res.items()]
    print(
        tabulate(
            table, headers=["Number of Negative Experiences", "mean dds", "median dds", "sd", "n"]
        )
    )
    res = data_set.correlation_neg_exp_dds()
    print("Correlation between dds and neg exp", res)

def print_neg_exp_dev_mech(data_set: Dataset):
    print_header("Correlation between Negative Experiences and usage of device mechanisms")
    res = data_set.device_mechanism_correlated_to_negative_experiences()
    print(tabulate(res.items(), headers=["Mechanism", "Correlation"]))

FILE = "data/data_smartphone-notifications.txt"

data_set = Dataset(FILE)
data_set.process()
print_header("Basics")
print(
    f"Total: {data_set.total_count}, Valid: {data_set.valid_count}, Percentage valid: {data_set.valid_count/data_set.total_count*100}%"
)
print(data_set.per_source(), data_set._cnt_to_perc(data_set.per_source()), data_set.drop_out_rates_by_source())
print_header("Age")
print_age(data_set)
print_header("Gender")
print_gender_distribution(data_set)
print_header("Nation")
print_nation(data_set)

print_education(data_set)
print_employment(data_set)

print_os(data_set)

print_number_of_negative_exp(data_set)
print_concern_family(data_set)
print_concern_colleague(data_set)
print_privacy_concern_wilcoxon(data_set)
print_concern_self(data_set)
print_analysis(
    "Last notification contained Information regarding another party",
    data_set._analyze_col(
        "A509",
        lambda v: YesNo(int(v)),
        lambda v: v != YesNo.NA and v != YesNo.SKIPPED and v != YesNo.OTHER,
    ),
)
print_concern_other(data_set)
print_mean_concern_family(data_set)
print_mean_conern_colleage(data_set)
print_device_sharing(data_set)
print_device_mechanism(data_set)

print_multi_col_analysis(
    "Times another person can read notifications",
    data_set.times_another_person_can_read_notifications_by_place(),
)
print_multi_col_analysis(
    "Acitions taken before sharing",
    data_set._analyse_multi_col(
        ColMappingSharingPrequesits, lambda v: Checkbox(int(v))
    ),
)
print_multi_col_analysis(
    "Clearing Notifications by persons handing the phone",
    data_set._analyse_multi_col(
        ColMappingClearNotifications,
        lambda v: Frequency(int(v)),
        lambda k: k != Frequency.NA and k != Frequency.SKIPPED,
    ),
)
print_multi_col_analysis(
    "Reason for bad experience",
    data_set._analyse_multi_col(
        ColMappingBadExperienceReason, lambda v: Checkbox(int(v))
    ),
)
print_neg_exp_dds(data_set)
print_neg_exp_surv_intr_dev_mech(data_set)
print_analysis("Control mechanisms used", data_set.device_mechanisms_used())
print_neg_exp_dev_mech(data_set)