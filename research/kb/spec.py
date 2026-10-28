"""Constants shared by the renderer and the validator.

The field order and the limits here are the contract in
docs/zh-material-spec.md. The training corpus, the on-device resource set and
the checker all have to agree, so they live in one place and everyone imports
this module rather than restating them.
"""

FIELDS = ("人物", "时间", "地点", "起因", "经过", "结果")

FIELD_MAX = {
    "人物": 40,
    "时间": 20,
    "地点": 20,
    "起因": 60,
    "经过": 100,
    "结果": 50,
}

MATERIAL_MAX = 300
STORY_MAX = 360
STORY_MIN = 120
TITLE_MAX = 24

MATERIAL_MARK = "<|material|>"
STORY_MARK = "<|story|>"
EOD_MARK = "<|endoftext|>"

# Surnames and compound surnames. The checker uses these to find proper nouns
# a material never introduced. Only personal names are listed in 人物 when the
# source has them; a role label ("图书管理员") is matched by substring instead,
# which is why this list is for the reverse check alone.
SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "谢邹喻柏窦章苏潘葛范彭鲁韦马苗方俞任袁柳唐薛雷贺倪汤罗毕郝安常"
    "傅康伍余顾孟黄穆萧尹姚邵汪毛米贝明戴宋庞熊纪舒屈项祝董梁杜阮蓝"
    "闵席季麻强贾路娄江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万卢"
    "莫房解应丁邓单杭洪包左石崔吉龚程邢裴陆荣翁荀惠甄曲封储靳焦巴班"
    "秋仲伊宫宁仇甘厉祖武符刘景詹龙叶黎白蒲从鄂索赖卓屠蒙池乔闻党翟"
    "谭姬申冉雍桑桂牛通边燕浦尚温庄晏柴阎慕连茹艾容向古易廖终衡步都"
    "耿满匡国文寇广东欧沃利蔚越师巩聂晁冷辛那简饶曾沙鞠关查荆红游权盖"
)

COMPOUND_SURNAMES = (
    "欧阳", "司马", "诸葛", "上官", "皇甫", "尉迟", "公孙", "慕容",
    "令狐", "长孙", "宇文", "司徒", "夏侯", "端木", "东方", "独孤",
    "南宫", "西门", "百里", "呼延", "澹台",
)

PLACE_TAILS = "镇村店桥山河寺庙馆铺巷街庄坊城池关港岛屿州府县郡"

# Names a corpus must never fall back on: if the model has a default name to
# reach for, it stops reading the material.
DEFAULT_NAMES = ("小明", "小红", "张三", "李四", "王五", "某某", "某人")

# Words that belong to the narration layer. In a material they turn claims
# into hearsay, and the model will repeat them as fact.
NARRATION_STRONG = ("据说", "听说", "我认为", "我觉得", "可能", "大概是")
NARRATION_WEAK = ("相传", "传说", "民间流传")

VERBS_AFTER_NAME = "说道问答走来去看想站坐拿放叫喊哭笑的把被在从对给让送递交给托"
