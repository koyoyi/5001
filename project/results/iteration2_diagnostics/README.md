# TechQA 训练与开发集检索失败诊断

开发集 84 个查询中，原候选并集漏检 14 个；全文 BM25 在 top100 找回其中 2 个。全文 BM25 的 nDCG@10 从 0.5997 变为 0.5960，Recall@20 从 0.6905 变为 0.6905。单纯改为全文没有带来本开发集的总体收益；全文输入的改善与退化均保留，不能据个案判断所有长文都受益。

## 比较范围

沿用原训练和开发划分、28,478 篇去重文档、BM25 参数和确定性的并列分数规则。原输入是每篇前 256 个空白分隔词；对照只把 BM25 文档改为全文。索引全文来自全部未标注语料，不按 gold 筛文档，不训练模型，不把答案加进特征或候选。原 SQLite 只查询 train/dev，测试指标及案例不参与本次诊断。训练结果包含拟合数据，应以开发结果决定下一步。查询ID中的TRAIN或DEV前缀来自源文件，不表示本次自定义划分；本轮仍非官方保留集评测。

语料中 22,050 篇超过 256 词；完整正文词数中位数 385、第95百分位 965。

## 候选与排名失败

| 划分 | 查询 | 原候选漏检 | gold已在候选但RRF未进top10 | Logistic未进top10 | RF未进top10 | BM25的nDCG高于RRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 412 | 66 | 97 | 75 | 9 | 128 |
| dev | 84 | 14 | 12 | 13 | 12 | 29 |

候选漏检无法由候选内重排修复。BM25优于RRF的查询说明当前LSA与固定融合会稀释强词项证据；学习器超过RRF不等于超过BM25。

## 文档输入对照

| 划分 | 输入 | nDCG@10 | Recall@20 | Recall@100 |
| --- | --- | ---: | ---: | ---: |
| train | 前256词 | 0.5428 | 0.7039 | 0.8180 |
| train | 全文 | 0.5683 | 0.7330 | 0.8544 |
| dev | 前256词 | 0.5997 | 0.6905 | 0.7976 |
| dev | 全文 | 0.5960 | 0.6905 | 0.7857 |

开发集有 3 个查询的 nDCG@10 提高、6 个降低，其余相同。全文改变词频和长度归一化，因此平均变化不能全部归因于答案被截断。下一步应把全文BM25、统一分块后按文档聚合的检索与原前缀基线并列，固定同一划分和完整文档集合，再选择输入策略。

开发集全文减前缀的 nDCG@10 差值为 -0.0036，按 72 个组件配对 bootstrap 的探索性95%区间为 [-0.0195, +0.0117]。该开发集用于后续方案选择，区间不能当作新的最终测试证据。

## 已知答案的位置

在原 gold 正文中查找源答案的大小写及空白归一化精确字符串，按空白词位置与第 256 词边界比较。此项使用答案只做人工错误诊断；找不到精确字符串的查询记为未知，不能据此认定没有证据。完整答案落在尾部能证明这一字符串不在前缀，但仍不能证明前缀完全没有可用线索。

| 划分 | 精确答案位置 | 查询 | 原候选漏检 | 前缀BM25 Recall@100 | 全文BM25 Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | 跨边界 | 25 | 5 | 0.7600 | 0.8000 |
| train | 前缀内 | 323 | 39 | 0.8638 | 0.8669 |
| train | 未精确定位 | 1 | 0 | 1.0000 | 1.0000 |
| train | 边界后 | 63 | 22 | 0.6032 | 0.8095 |
| dev | 跨边界 | 9 | 4 | 0.5556 | 0.5556 |
| dev | 前缀内 | 66 | 7 | 0.8485 | 0.8182 |
| dev | 边界后 | 9 | 3 | 0.6667 | 0.7778 |

## 下一轮比较重点

优先检查固定融合为何降低强词项命中的名次。技术实体、CVE编号、版本号及错误码的一致性都可以从查询与文档直接计算，适合考虑作为可解释特征；这些建议需要训练组件CV支持，不能按开发案例逐条添加例外。BM25必须继续作为单独的强基线。

长文分块应固定窗口、重叠和文档聚合规则，对所有方法使用一致的可见正文，并把块映射回原文档ID计算指标。若把前缀与全文或分块检索合并，必须重新报告候选规模和漏检率，所有重排方法排序同一完整并集。源答案及本次位置标注只用于诊断，不应转化为检索或训练时可见的字段。

## 代表案例

每类最多选三个，优先开发集；先按该类量化差异排序，再按查询ID确定并列顺序。案例仅用于解释，不代表随机样本。

### 案例 1 原候选漏检被全文BM25找回

查询 `TRAIN_Q429`，dev：Does Portal 7.0.0.2 CF20 support upgrading the DB2 JDBC Driver from 4.11.77 to 4.18.60? Does Portal 7.0.0.2 CF20 support upgrading the DB2 JDBC Driver from 4.11.77 to 4.18.60?

gold：`swg27020885.txt`。前缀BM25 gold名次 未进已记录范围，全文BM25 25，原RRF top20内名次 未进已记录范围；原候选召回 0。源答案位置：边界后，起始空白词序号 2184（从0计）。

源答案摘录：IBM DB2 Universal JDBC Provider (XA) from DB2 z/OS (type 4) Version and future fix packs All Applicable AIX OS Requires APAR PK93123 for DB2 on z/OS V8.1 and V9.1. You can obtain the fix from DB2 on z/OS support or IBMLINK.

gold正文相关位置摘录：(type 2/4) 4.0 IBM Application Connectivity to DB2 for z/OS Feature Version and future fix packs All Applicable AIX OS IBM DB2 Universal JDBC Provider (XA) from DB2 z/OS (type 4) Version and future fix packs All Applicable AIX OS Requires APAR PK93123 for DB2 on z/OS V8.1 and V9.1. You can obtain the fix from DB2 on z/OS support or IBMLINK. IBM Toolbox for Java/JTOpen JDBC Driver 6.0 and future fix packs All Applicable AIX OS Check for the latest jt400.jar file at:http://sourceforge.net/projects/jt400/files/ [http://sourceforge.net/projects/jt400/files/]

### 案例 2 原候选漏检被全文BM25找回

查询 `TRAIN_Q066`，dev：Is there a 64-bit agent Log File Agent available for Windows OS? Is there a 64-bit agent Log File Agent available for Windows OS?

gold：`swg24043263.txt`。前缀BM25 gold名次 未进已记录范围，全文BM25 71，原RRF top20内名次 未进已记录范围；原候选召回 0。源答案位置：前缀内，起始空白词序号 19（从0计）。

源答案摘录：This fix pack adds support for a Windows x86-64 bit (WIX64) and PowerPC little-endian Log File agent.

gold正文相关位置摘录：Title: IBM Tivoli Log File Agent, Version 6.3.0 Fix Pack 02 (6.3.0-TIV-ITM_LFA-FP0002) - United States Text: DOWNLOADABLE FILES ABSTRACT This fix pack adds support for a Windows x86-64 bit (WIX64) and PowerPC little-endian Log File agent. In addition, it resolves the APARs and defects listed below. DOWNLOAD DESCRIPTION (C) Copyright International Business Machines Corporation 2017. All rights reserved. Component: Tivoli® Log File Agent, Version 6.3.0 Component ID: 5724C04LF Fix Pack: 0002, 6.3.0-TIV-ITM_LFA-FP0002 Date: September 19, 2017 Contents: 1.0 General description 2.0 Problems fixed 3.0 Architecture

### 案例 3 原候选漏检被全文BM25找回

查询 `DEV_Q202`，train：Why is installation manager cores when try to install netcool using GUI mode in AIX 7.1? IBM Installation Manager Cores when running in GUI mode on AIX 7.1

gold：`swg21631478.txt`。前缀BM25 gold名次 未进已记录范围，全文BM25 7，原RRF top20内名次 未进已记录范围；原候选召回 0。源答案位置：边界后，起始空白词序号 743（从0计）。

源答案摘录：To use the GUI, the gtk+ RPM and dependencies must be available on AIX. Install this set of RPMs on AIX:

gold正文相关位置摘录：all packages. For console mode, check your product documentation to verify that the product supports console mode. RESOLVING THE PROBLEM To use the GUI, the gtk+ RPM and dependencies must be available on AIX. Install this set of RPMs on AIX: atk-1.12.3-2.aix5.2.ppc.rpm cairo-1.8.8-1.aix5.2.ppc.rpm expat-2.0.1-1.aix5.2.ppc.rpm fontconfig-2.4.2-1.aix5.2.ppc.rpm freetype2-2.3.9-1.aix5.2.ppc.rpm gettext-0.10.40-6.aix5.1.ppc.rpm glib2-2.12.4-2.aix5.2.ppc.rpm gtk2-2.10.6-4.aix5.2.ppc.rpm libjpeg-6b-6.aix5.1.ppc.rpm libpng-1.2.32-2.aix5.2.ppc.rpm libtiff-3.8.2-1.aix5.2.ppc.rpm pango-1.14.5-4.aix5.2.ppc.rpm pixman-0.12.0-3.aix5.2.ppc.rpm xcursor-1.1.7-3.aix5.2.ppc.rpm xft-2.1.6-5.aix5.1.ppc.rpm xrender-0.9.1-3.aix5.2.ppc.rpm zlib-1.2.3-3.aix5.1.ppc.rpm The installed RPMs must be of the minimum versions listed above. To install the RPM versions specified above take the following steps: 1. Download the gtk2_bundle_v2.tar package

### 案例 4 尾部答案与全文BM25改善

查询 `TRAIN_Q546`，dev：Does @LocalNodeAlias="" will affect ITNM failover health check events?. Customization in nco_p_monitor.rules with @LocalNodeAlias="empty", after that both Primary and Backup started health check fail event, how to resolve that.

gold：`swg21424808.txt`。前缀BM25 gold名次 4，全文BM25 3，原RRF top20内名次 6；原候选召回 1。源答案位置：跨边界，起始空白词序号 209（从0计）。

源答案摘录：The value for m_Domain is taken from the value for Node (@Node) in alerts.status for the health check event. If you have altered the inserted value for @Node, either by modifying the nco_p_ncpmonitor.rules file or by some other method such as an Automation or Impact policy, then the resulting update for state.domains will fail. Ensure that the rules governing the modification of @Node allow for the value of @Node to remain unchanged for health check events.

gold正文相关位置摘录：It should not be the name of the Virtual domain. ProcessHealthCheck sent:update state.domains set m_HealthStatus=1 where m_Domain='ITNM'; RESOLVING THE PROBLEM The value for m_Domain is taken from the value for Node (@Node) in alerts.status for the health check event. If you have altered the inserted value for @Node, either by modifying the nco_p_ncpmonitor.rules file or by some other method such as an Automation or Impact policy, then the resulting update for state.domains will fail. Ensure that the rules governing the modification of @Node allow for

### 案例 5 尾部答案与全文BM25改善

查询 `DEV_Q080`，train：Why is the OUTPUT_TYPE specified in the properties file for the custom scripting feature ignored? Why is the OUTPUT_TYPE specified in the properties file for the custom scripting feature ignored in ITM V6? Specifying "I" for INTEGER, or "F" for FLOAT for the OUTPUT_TYPE doesn't work, but specifying OUTPUT_TYPE=S results in STRING type for the fi...

gold：`swg21960062.txt`。前缀BM25 gold名次 13，全文BM25 1，原RRF top20内名次 未进已记录范围；原候选召回 1。源答案位置：边界后，起始空白词序号 2852（从0计）。

源答案摘录：OUTPUT_TYPE (STRING|INTEGER|FLOAT - Optional - default string). It defines the type of the first token returned by each row of the script;

gold正文相关位置摘录：tokens for a maximum of 5 strings, 5 integers and 5 floats following a predefined syntax as described below: - OUTPUT_TYPE (STRING|INTEGER|FLOAT - Optional - default string). It defines the type of the first token returned by each row of the script; OUTPUT_TYPE can be: 1. STRING (default): strings up to 2048 characters. When used, the "Standard_Output_String" attribute of KXX_Custom_Scripts_Runtime is filled in by the first token. 2. INTEGER: allows getting numeric values between -9223372036854775806 and 9223372036854775806. When used, the "Standard_Output_Integer" attribute of KXX_Custom_Scripts_Runtime is

### 案例 6 尾部答案与全文BM25改善

查询 `DEV_Q153`，train：Can we create Match design specification db in xmeta Could you please let me know if we can create schema for Match design specification database in xmeta db.

gold：`swg21409481.txt`。前缀BM25 gold名次 40，全文BM25 1，原RRF top20内名次 未进已记录范围；原候选召回 1。源答案位置：边界后，起始空白词序号 378（从0计）。

源答案摘录：You should start from an empty Database; using the default XMETA Database is not recommended because it can become corrupt.

gold正文相关位置摘录：the most accurate matching thresholds a full volume run should be performed in the DataStage Designer. Creating the Database 1. You should start from an empty Database; using the default XMETA Database is not recommended because it can become corrupt. 2. There are certain requirements on how the Database is to be created. a) If Oracle: CHARACTERSET should be set to AL32UTF b) If SQL: default is fine. c) If DB2: create as codepage UTF-8 and page size 32K d) Permission’s - Each user has

### 案例 7 原BM25优于RRF

查询 `TRAIN_Q171`，dev：Help with Security Bulletin: Security vulnerabilities have been identified in IBM DB2 shipped with WebSphere Remote Server (CVE-2014-8730) I need to understand details regarding Security Bulletin: Security vulnerabilities have been identified in IBM DB2 shipped with WebSphere Remote Server (CVE-2014-8730). Where can I find this information?

gold：`swg21700633.txt`。前缀BM25 gold名次 1，全文BM25 1，原RRF top20内名次 8；原候选召回 1。源答案位置：前缀内，起始空白词序号 52（从0计）。

源答案摘录：For vulnerability details, see the security bulletin TLS padding vulnerability affects IBM DB2 LUW (CVE-2014-8730)

gold正文相关位置摘录：of WebSphere Remote Server. Information about security vulnerabilities affecting IBM DB2 has been published in a security bulletin. VULNERABILITY DETAILS For vulnerability details, see the security bulletin TLS padding vulnerability affects IBM DB2 LUW (CVE-2014-8730) [http://www.ibm.com/support/docview.wss?uid=swg21692618]. AFFECTED PRODUCTS AND VERSIONS Principal Product and Version(s) Affected Supporting Product and Version WebSphere Remote Server V6.1, 6.2, 6.2.1, 7.0, 7.1, 7.1.1, 7.1.2, 8.5 IBM DB2 Workgroup Server Edition V9.1, 9.5, 9.7, 10.1, 10.5 GET NOTIFIED ABOUT FUTURE SECURITY BULLETINS Subscribe to My Notifications [ http://www-01.ibm.com/software/support/einfo.html] to be notified

### 案例 8 原BM25优于RRF

查询 `TRAIN_Q084`，dev：Help with Security Bulletin: IBM MQ Clients connecting to an MQ Queue Manager can cause a SIGSEGV in the AMQRMPPA channel process terminating it (CVE-2018-1371) I need to understand details regarding Security Bulletin: IBM MQ Clients connecting to an MQ Queue Manager can cause a SIGSEGV in the AMQRMPPA channel process terminating it (CVE-2018-13...

gold：`swg22012983.txt`。前缀BM25 gold名次 1，全文BM25 1，原RRF top20内名次 6；原候选召回 1。源答案位置：前缀内，起始空白词序号 57（从0计）。

源答案摘录：CVEID: CVE-2018-1371 [http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2018-1371] DESCRIPTION: An IBM MQ client connecting to an MQ queue manager can cause a SIGSEGV in the amqrmppa channel process terminating it. CVSS Base Score: 6.5 CVSS Temporal Score: See https://exchange.xforce.ibmcloud.com/vulnerabilities/137771 [https://exchange.xforce.ibmcloud.com/vulnerabilities/137771] for the current score CVSS Environmental Score*: Undefined CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H

gold正文相关位置摘录：to an MQ queue manager can cause a SIGSEGV in the queue manager's amqrmppa channel process, terminating it. VULNERABILITY DETAILS CVEID: CVE-2018-1371 [http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2018-1371] DESCRIPTION: An IBM MQ client connecting to an MQ queue manager can cause a SIGSEGV in the amqrmppa channel process terminating it. CVSS Base Score: 6.5 CVSS Temporal Score: See https://exchange.xforce.ibmcloud.com/vulnerabilities/137771 [https://exchange.xforce.ibmcloud.com/vulnerabilities/137771] for the current score CVSS Environmental Score*: Undefined CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H) AFFECTED PRODUCTS AND VERSIONS IBM MQ V8.0 * Maintenance level 8.0.0.8 IBM MQ V9 LTS * Maintenance level 9.0.0.2

### 案例 9 原BM25优于RRF

查询 `TRAIN_Q144`，dev：Help with Security Bulletin: The WebAdmin context for WMB V8 allows directory listings (CVE-2016-6080) I need to understand details regarding Security Bulletin: The WebAdmin context for WebSphere Message Broker Version 8 allows directory listings (CVE-2016-6080). Where can I find this information? 0

gold：`swg21995004.txt`。前缀BM25 gold名次 1，全文BM25 1，原RRF top20内名次 6；原候选召回 1。源答案位置：前缀内，起始空白词序号 40（从0计）。

源答案摘录：CVEID: CVE-2016-6080 [http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2016-6080] DESCRIPTION: The WebAdmin context for WebSphere Message Broker allows directory listings, which could disclose sensitive information to the attacker. CVSS Base Score: 5.3 CVSS Temporal Score: See https://exchange.xforce.ibmcloud.com/vulnerabilities/117560 [https://exchange.xforce.ibmcloud.com/vulnerabilities/117560] for the current score CVSS Environmental Score*: Undefined CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:N/UI

gold正文相关位置摘录：Text: WMB IIB SECURITY BULLETIN SUMMARY The WebAdmin context for WebSphere Message Broker Version 8 allows directory listings. VULNERABILITY DETAILS CVEID: CVE-2016-6080 [http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2016-6080] DESCRIPTION: The WebAdmin context for WebSphere Message Broker allows directory listings, which could disclose sensitive information to the attacker. CVSS Base Score: 5.3 CVSS Temporal Score: See https://exchange.xforce.ibmcloud.com/vulnerabilities/117560 [https://exchange.xforce.ibmcloud.com/vulnerabilities/117560] for the current score CVSS Environmental Score*: Undefined CVSS Vector: (CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N) AFFECTED PRODUCTS AND VERSIONS WebSphere Message Broker V8 REMEDIATION/FIXES Product VRMF APAR Remediation/Fix WebSphere Message Broker V8 IT16698 The APAR is available

### 案例 10 全文BM25退化

查询 `DEV_Q046`，dev：Cannot delete folder in Box collab folder In an ICM 5.3.0 environment integrated with Box, after setting up Box Collaboration, case worker can create Box collaboration folder, and can create folders and files under Box collaboration folder, however, they cannot delete the their folders and files. Case worker's default permission is 'Viewer Uploa...

gold：`swg22002589.txt`。前缀BM25 gold名次 1，全文BM25 2，原RRF top20内名次 1；原候选召回 1。源答案位置：前缀内，起始空白词序号 57（从0计）。

源答案摘录：There is no support for deleting documents and/or folders from a Box Collaboration folder created via Case Manager at this time.

gold正文相关位置摘录：folders created under the Box collaboration folder. There does not appear to be any mechanism to perform this deletion. ANSWER There is no support for deleting documents and/or folders from a Box Collaboration folder created via Case Manager at this time. Case workers who created the Box collaboration folder will have Viewer/Uploader permission on the Box Collaboration folder, and therefore, will not be able to delete content within a Box collaboration folder as this may result in missing data in the associated cases. Users who

### 案例 11 全文BM25退化

查询 `TRAIN_Q301`，dev：How do you switch on the indexing of sparse file in StoredIQ? By default these are skipped - does anyone know where to switch them on on the dataserver?

gold：`swg21692240.txt`。前缀BM25 gold名次 2，全文BM25 6，原RRF top20内名次 1；原候选召回 1。源答案位置：前缀内，起始空白词序号 65（从0计）。

源答案摘录：SSH into the Data Server(s) and enter the following command at the prompt; psql -U dfuser dfdata -c "insert into adminknobs (name,value,description,valuetype,use) VALUES ('read_sparse_files','1','Read sprase data objects','int','1')" It should return the following; INSERT 0 1 if so, enter the following command and wait for the UI to come back up; service siqinit restart

gold正文相关位置摘录：a sparse data object message recorded in the details section. To correct this problem, ensure that no jobs are running. SSH into the Data Server(s) and enter the following command at the prompt; psql -U dfuser dfdata -c "insert into adminknobs (name,value,description,valuetype,use) VALUES ('read_sparse_files','1','Read sprase data objects','int','1')" It should return the following; INSERT 0 1 if so, enter the following command and wait for the UI to come back up; service siqinit restart You can now run harvests as normal and sparse data objects will

### 案例 12 全文BM25退化

查询 `TRAIN_Q091`，dev：Keys couldn't be imported. Unable to encrypt the FIPS key On windows10 machine when I attempt to import the encryption key I am getting the error: "Keys couldn't be imported. Unable to encrypt the FIPS key". Because I cannot import the keys, I am unable to validate parameters.

gold：`swg21648986.txt`。前缀BM25 gold名次 2，全文BM25 3，原RRF top20内名次 1；原候选召回 1。源答案位置：前缀内，起始空白词序号 73（从0计）。

源答案摘录：To allow the keys to be exported properly: 1. Select Local Security Policy under Administrative tools 2. Navigate to Local Policies - Security Options 3. Select System Cryptography: Use FIPS compliant algorithms for encryption, hashing and signing and be sure it is Disabled 4. Run dcskey e again to export the key

gold正文相关位置摘录：Enabling the Use FIPS compliant algorithms for encryption, hashing and signing security policy can cause this error RESOLVING THE PROBLEM To allow the keys to be exported properly: 1. Select Local Security Policy under Administrative tools 2. Navigate to Local Policies - Security Options 3. Select System Cryptography: Use FIPS compliant algorithms for encryption, hashing and signing and be sure it is Disabled 4. Run dcskey e again to export the key

## 复核材料

[汇总与来源哈希](summary.json)、[全部训练开发查询诊断](per_query_diagnostics.csv)、[最多十二个案例](representative_cases.json)。全文索引耗时是本机离线构建成本，不能作为在线查询延迟。
