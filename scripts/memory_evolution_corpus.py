"""Deterministically expand the Chinese memory benchmark with synthetic profiles."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from random import Random
from typing import NamedTuple


class Topic(NamedTuple):
    name: str
    values: str
    questions: tuple[str, ...]
    nearby: str


TOPICS = {
    "debug": (
        Topic("桌面壁纸", "薄雾山谷 浅滩贝壳 星轨照片 蓝鲸插画 梧桐落叶 老城剪影",
              ("桌面壁纸换哪张？", "桌面背景按我平时的习惯来吧。", "电脑开机那张图，我一般挑哪一种？"),
              "桌面壁纸设成动态的会多耗电吗？"),
        Topic("香薰气味", "白茶 柚木 广藿香 海盐 雪松 橙花",
              ("香薰气味选哪种？", "香薰补充装照旧买哪款？", "扩香机的精油，之前我爱用哪瓶？"),
              "香薰气味一般多久才会散干净？"),
        Topic("练字字体", "小楷 行书 隶书 瘦金体 魏碑 草书",
              ("练字字体用哪套？", "我练字时最常临哪种字体？", "选本适合我的字帖，平时我临哪体？"),
              "练字字体里的笔画名称是谁定的？"),
        Topic("阳台盆栽", "薄荷 茉莉 铃兰 矾根 迷迭香 虎尾兰",
              ("阳台盆栽种哪种？", "我的阳台盆栽通常养什么？", "栏杆边腾了个花盆，我平时想种啥？"),
              "阳台盆栽的陶盆碎了能回收吗？"),
        Topic("笔记版式", "康奈尔式 思维导图 分栏提纲 流程图 双页索引 时间轴",
              ("笔记版式怎么选？", "笔记版式按我之前用的来。", "整理这一页记录，用我顺手的哪套排法？"),
              "笔记版式可以在打印时一页放两份吗？"),
        Topic("拍照题材", "街边招牌 花窗 旧铁路 城市倒影 水鸟 人行天桥",
              ("拍照题材挑哪个？", "拍照题材我一般想拍什么？", "相机充好了，我外出取景更喜欢找什么？"),
              "拍照题材会影响照片的 EXIF 字段吗？"),
    ),
    "holdout": (
        Topic("茶饮茶底", "岩茶 碧螺春 白牡丹 蒙顶黄芽 六堡茶 滇红",
              ("茶饮茶底选什么？", "给我挑一款合口味的茶底。", "冲杯茶吧，我平时爱喝的叶子是哪种？"),
              "茶饮茶底过滤后留下的渣怎么处理？"),
        Topic("桌游类型", "合作解谜 轻量推理 工人放置 牌库构筑 竞速闯关 区域控制",
              ("桌游类型按我喜欢的选。", "我和朋友开桌游，一般选什么玩法？", "今天大家聚一起，我倾向开哪类局？"),
              "桌游类型通常印在包装盒的哪个面？"),
        Topic("编织线材", "牦牛绒 丝光棉 亚麻线 马海毛 彩点羊毛 竹纤维",
              ("编织线材用我习惯的。", "编织线材下次买哪种？", "要起针了，平时我更想从哪团毛线开始？"),
              "编织线材的标签该扔哪个垃圾桶？"),
        Topic("键盘轴体", "红轴 茶轴 青轴 银轴 静音灰轴 段落白轴",
              ("键盘轴体按我平时的来。", "键盘轴体要装哪种？", "换一套机械开关，我常用哪个手感？"),
              "键盘轴体拔下来之后能不能水洗？"),
        Topic("拼图图案", "古镇地图 海底珊瑚 复古邮票 沙漠星空 暴雨列车 玻璃温室",
              ("拼图图案选哪幅？", "我常买哪类图案的拼图？", "这次想拼个千片的，我平时挑什么画面？"),
              "拼图图案相同的两盒能混在一起拼吗？"),
        Topic("烘焙点心", "肉桂卷 杏仁饼干 黑芝麻贝果 柠檬塔 栗子玛芬 葡萄干司康",
              ("烘焙点心做哪款？", "这回的烘焙点心按我口味选。", "预热好烤箱了，我平常最想做什么？"),
              "烘焙点心用的硅胶垫能进洗碗机吗？"),
        Topic("画画工具", "油画棒 自动铅笔 水彩墨水 酒精马克笔 粉彩条 炭笔",
              ("画画工具用我喜欢的哪件？", "画画工具我最顺手的是什么？", "想涂几笔，我平时会从笔盒里拿什么？"),
              "画画工具收进纸盒前要晾多久？"),
        Topic("观影片种", "黑色喜剧 自然纪录片 公路电影 民俗奇幻 无声短片 悬疑剧情片",
              ("观影片种按我口味挑。", "下次观影选我常看的哪种片？", "开投影了，我通常喜欢看哪一类片子？"),
              "观影片种的英文标注在哪里能看到？"),
        Topic("杯子材质", "白瓷 粗陶 玻璃 搪瓷 不锈钢 钛合金",
              ("杯子材质买哪种？", "我平时用什么材质的杯子？", "想换个喝水的杯，我惯用的手感是哪种？"),
              "杯子材质会改变微波炉加热的速度吗？"),
        Topic("瑜伽课程", "阴瑜伽 流瑜伽 修复拉伸 呼吸冥想 核心平衡 肩颈舒展",
              ("瑜伽课程按我爱上的选。", "瑜伽课程我该约哪堂？", "铺好垫子了，我平时想练哪个课？"),
              "瑜伽课程改签后原来的签到码还有效吗？"),
        Topic("手工印章", "蕨叶纹 月相图 回纹 几何窗花 流水纹 雨燕剪影",
              ("手工印章我常刻哪种？", "手工印章的纹样帮我按老习惯挑。", "给这张卡盖图案，我一般想刻哪一种？"),
              "手工印章沾错颜色后怎么清洁？"),
        Topic("电脑配色", "雾蓝 松针绿 珊瑚红 石板灰 薰衣草紫 暖沙金",
              ("电脑配色用我喜欢的。", "电脑配色方案选哪个？", "窗口边框换个颜色，我之前偏向什么色？"),
              "电脑配色存在哪里，能导出吗？"),
        Topic("便当主食", "玉米糙米饭 紫米团 荞麦饼 杂豆卷 芝麻花卷 蒸山药",
              ("便当主食装哪样？", "便当主食按我平常喜欢的备。", "准备带饭了，饭盒里我平时会放什么主食？"),
              "便当主食放凉之前能扣紧饭盒盖吗？"),
        Topic("植物标本", "银杏叶 蕨芽 桂花枝 蒲公英 车前草 三叶草",
              ("植物标本夹哪一种？", "植物标本我通常收哪样？", "要压一片做书签，我平时更爱找什么？"),
              "植物标本做好后隔几天才可以装袋？"),
        Topic("骑行路线", "河堤绿道 绕湖平路 山脚缓坡 老工业区 果园窄道 大学环线",
              ("骑行路线按我平时的选。", "骑行路线我通常走哪一条？", "单车打好气了，照我口味往哪边骑？"),
              "骑行路线的里程标会每年更新吗？"),
        Topic("闹钟铃声", "竹笛 海浪 爵士鼓 木琴 鸟鸣 低频风铃",
              ("闹钟铃声设哪种？", "闹钟铃声帮我选我常用的。", "明天叫醒我的声音，我平时喜欢听哪种？"),
              "闹钟铃声在手机静音时还能响吗？"),
        Topic("文件命名", "日期优先 项目编号 主题短词 版本后缀 首字母索引 人类可读标题",
              ("文件命名按我习惯的来。", "文件命名规则我最常用哪套？", "整理这一批附件，我以前是怎么给它们起名的？"),
              "文件命名可以保留空格再批量打包吗？"),
        Topic("客厅照明", "暖白射灯 落地纸灯 琥珀灯带 反射顶灯 壁挂阅读灯 可调轨道灯",
              ("客厅照明开哪套？", "客厅照明按我喜欢的布置。", "沙发旁边我平时倾向亮哪种光？"),
              "客厅照明装了两个开关还能接调光器吗？"),
    ),
}

UNRELATED = {
    "debug": (
        "勺柄上的刻度怎么看？", "轮胎侧面的字母是什么意思？", "几号是下一个农历十五？",
        "铝箔纸两面颜色为什么不同？", "快递柜的取件码过期会怎么样？", "怎样把一张纸折成五等份？",
        "门锁的电池能撑多久？", "三角尺上有哪几个角度？", "水壶里那层白垢是什么？",
    ),
    "holdout": (
        "明早有没有潮汐？", "一节地铁车厢一般有几个门？", "冰块融化前后重量相同吗？",
        "地球的磁极多久移动一次？", "这周六是几月几号？", "拉链卡在布上怎么拉开？",
        "邮局什么时候停止收当日包裹？", "拼音里的轻声有声调符号吗？", "玻璃上的贴纸能泡软吗？",
    ),
}

RECORDS = (
    "说到{topic}，我一般选{value}。",
    "{topic}给我记一下，我一直用{value}。",
    "好几次选下来，{topic}还是{value}最合我的口味。",
    "要是问我{topic}喜欢哪种，答案通常是{value}。",
    "最近有人问起{topic}，我才发现平常更喜欢{value}。",
    "帮我把{topic}记成{value}吧，这是平时的选择。",
)
CHANGES = (
    "以后{topic}改为{value}，原先的{old}就不选了。",
    "{old}用腻了。说到{topic}，往后按{value}来。",
    "更新一下长期记录，{topic}现在固定是{value}。",
    "刚才重新试过了，今后{topic}我想选{value}。",
)
TEMPORARY = (
    "{topic}这次用{value}，接下来{hours}小时有效。",
    "临时换一下{topic}。到{until}之前先用{value}。",
    "现在先把{topic}改成{value}，到{until}就恢复平时那样。",
)
SCENE_RECORDS = (
    "我对{topic}分白天和晚上。白天选{day}，晚上选{night}。",
    "{topic}看时间。晚间我更常用{night}，日间是{day}。",
    "记下两种{topic}的习惯吧，白天{day}，晚间{night}。",
)
FAMILIES = {
    "unrelated": "重启后询问无关问题",
    "correction": "再次询问更正后的偏好",
    "expiry": "例外过期后询问",
    "scene": "按场景和用户复用",
}
BASE_DATE = datetime(2026, 4, 1, tzinfo=timezone(timedelta(hours=8)))


def _step(case_id, step_id, label, at, text, *, scene="daytime", expected=None,
          forbidden=(), writes=(), **annotations):
    return {
        "id": step_id, "label": label, "at": at.isoformat(), "scene": scene, "text": text,
        "expected": expected, "forbidden": list(forbidden), "writes": list(writes),
        "session_id": f"{case_id}-{step_id}", **annotations,
    }


def _profile(split, index):
    catalog = TOPICS[split]
    variant, topic_index = divmod(index, len(catalog))
    topic = catalog[topic_index]
    values = topic.values.split()
    round_index, offset = divmod(variant, len(values))
    # Rotate both replacements across the full range before reusing either one.
    replacement_count = len(values) - 1
    change_offset = round_index % replacement_count + 1
    temporary_offset = (change_offset + round_index // replacement_count) % replacement_count + 1
    old, changed, temporary = (values[(offset + shift) % len(values)]
                               for shift in (0, change_offset, temporary_offset))
    rng = Random(f"memory-evolution-corpus-v1/{split}/{index}")
    uid = f"expanded-{split}-{index + 1:05d}"
    predicate = f"{topic.name}偏好"
    at = BASE_DATE + timedelta(days=rng.randrange(210), hours=rng.randrange(7, 17),
                               minutes=rng.randrange(60))
    restart = at + timedelta(hours=rng.randrange(9, 170), minutes=rng.randrange(60))
    after_restart = restart + timedelta(hours=rng.randrange(4, 160), minutes=rng.randrange(60))
    short_hours = rng.choice((2, 4, 6, 12, 20, 36))
    expires = restart + timedelta(hours=short_hours)
    active = restart + timedelta(minutes=rng.randrange(1, short_hours * 60))
    expired = expires + timedelta(minutes=rng.randrange(0, 2881))
    question, second_question = rng.sample(topic.questions, 2)
    record = RECORDS[(round_index + offset + topic_index) % len(RECORDS)]
    record_text = record.format(topic=topic.name, value=old)

    def write(value, scene="default", valid_to=None):
        return {
            "predicate": predicate, "value": value, "content": f"{topic.name}习惯选{value}",
            "scene": scene, "valid_to": valid_to.isoformat() if valid_to else None,
        }

    def case(family, steps):
        return {
            "id": f"{family}-{uid}", "split": split, "user_id": uid,
            "title": f"{topic.name} · {FAMILIES[family]}",
            "topic_family": topic.name, "steps": steps,
        }

    unrelated_id = f"unrelated-{uid}"
    negatives = (topic.nearby, *UNRELATED[split])
    # Include topical questions with no need for the preference, in addition to
    # distant questions. A rotating prompt gives each value different negative probes.
    unrelated_text = negatives[(round_index + topic_index) % len(negatives)]
    yield case("unrelated", [
        _step(unrelated_id, "record", "写入原始偏好", at, record_text, writes=[write(old)]),
        _step(unrelated_id, "probe", "重启后询问无关问题", restart, unrelated_text,
              expected=[], unrelated=True),
    ])

    correction_id = f"correction-{uid}"
    change_text = rng.choice(CHANGES).format(topic=topic.name, value=changed, old=old)
    yield case("correction", [
        _step(correction_id, "record", "写入原始偏好", at, record_text, writes=[write(old)]),
        _step(correction_id, "change", "确认长期更正", restart, change_text, writes=[write(changed)]),
        _step(correction_id, "probe", "再次询问更正后的偏好", after_restart, question,
              expected=[changed], forbidden=[old], stale=[old]),
    ])

    expiry_id = f"expiry-{uid}"
    deadline = f"{expires.month}月{expires.day}日{expires.hour:02d}:{expires.minute:02d}"
    temporary_text = rng.choice(TEMPORARY).format(topic=topic.name, value=temporary,
                                                  hours=short_hours, until=deadline)
    yield case("expiry", [
        _step(expiry_id, "record", "写入原始偏好", at, record_text, writes=[write(old)]),
        _step(expiry_id, "temporary", "写入限时例外", restart, temporary_text,
              writes=[write(temporary, valid_to=expires)]),
        _step(expiry_id, "active", "例外有效时询问", active, question,
              expected=[temporary], forbidden=[old]),
        _step(expiry_id, "probe", "例外过期后询问", expired, second_question,
              expected=[old], forbidden=[temporary], stale=[temporary]),
    ])

    scene_id = f"scene-{uid}"
    day, night = "daytime", "evening_return"
    recorded_scenes = rng.choice(SCENE_RECORDS).format(topic=topic.name, day=old, night=changed)
    other_scene, other_value = (day, "白天") if variant % 2 else (night, "晚上")
    other_text = f"{other_value}说到{topic.name}，我一般选{temporary}。"
    evening = restart.replace(hour=rng.randrange(18, 23))
    following_day = evening.replace(hour=rng.randrange(8, 16)) + timedelta(days=rng.randrange(1, 6))
    yield case("scene", [
        _step(scene_id, "record", "写入日间和晚间偏好", at, recorded_scenes,
              writes=[write(old, day), write(changed, night)]),
        _step(scene_id, "other-user", "同库另一位虚构用户", at + timedelta(minutes=11), other_text,
              scene=other_scene, writes=[write(temporary, other_scene)], user_id=f"other-{uid}"),
        _step(scene_id, "probe", "晚间按场景复用", evening, f"晚上。{question}", scene=night,
              expected=[changed], forbidden=[old, temporary]),
        _step(scene_id, "day", "白天按场景复用", following_day, f"白天。{second_question}", scene=day,
              expected=[old], forbidden=[changed, temporary]),
    ])


def scale_dataset(dataset: dict, multiplier: int) -> dict:
    """Expand a grouped benchmark while retaining its original trajectories and demo."""
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError("语料倍数必须是正整数")
    if multiplier == 1:
        return dataset

    cases = dataset["cases"]
    counts = Counter((case["split"], case["id"].split("-", 1)[0]) for case in cases)
    profiles = {split: counts[split, "unrelated"] for split in TOPICS}
    expected = {(split, family): count for split, count in profiles.items() for family in FAMILIES}
    if counts != expected or not all(profiles.values()):
        raise ValueError("扩充语料需要按用户分组的四类基础轨迹")

    expanded = list(cases)
    for split, count in profiles.items():
        for index in range(count * (multiplier - 1)):
            expanded.extend(_profile(split, index))
    if len({case["id"] for case in expanded}) != len(expanded):
        raise ValueError("扩充后的轨迹 ID 重复")
    return {
        **dataset,
        "version": f"{dataset['version']}-synthetic-x{multiplier}",
        "split_rule": dataset["split_rule"] + "；新增虚构用户及主题按调试集和留出集隔离",
        "cases": expanded,
        "multiplier": multiplier,
    }
