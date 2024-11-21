# Know Your Customer框架

## 前言

KYC 是“了解你的客户”（Know Your Customer）或“了解你的客户端”（Know Your Client）的缩写。企业采用 KYC 检查来确认客户的身份，并持续评估和监控与之相关的任何风险。

## 需要进行KYC的组织

KYC（了解你的客户）是某些组织的强制性法律要求，主要是金融行业的组织。然而，这些要求因国家而异。金融服务中进行KYC检查的目的是限制洗钱、恐怖融资、腐败和其他非法活动。

主要流程：
- 客户身份识别程序（CIP）
- 客户尽职调查（CDD）

我们的声音识别主要用于客户身份识别程序

## 需要KYC的原因

于金融机构来说，根据法律和法规验证客户身份是法律要求。这包括反洗钱（AML）法律。

KYC要求因地域而异，因此检查当地法规很重要。在欧洲，与KYC最相关的两项法规是GDPR和AML5指令（或5AMLD）。企业还会希望熟悉eIDAS法规。

但个别国家也可以施加自己的额外要求。在德国，机构必须作为客户身份验证的一部分实施视频KYC流程。西班牙要求增强活体检测，法国需要第二份身份证件，意大利需要七项额外的风险检查。

在美国，金融犯罪执法网络（FinCEN）是主要的AML监管机构。《银行保密法》（BSA）是最重要的反洗钱法律。《美国爱国者法案》针对与恐怖主义有关的金融犯罪。美国也是金融行动特别工作组（FATF）的成员。

不遵守KYC/AML法律和法规可能会产生严重后果。最严重的违规行为可能会导致罚款和监禁。反洗钱和KYC合规不严是导致罚款的最常见问题之一。例如，澳大利亚的Westpac银行因AML违规被罚款9亿美元。

## KYC流程

包括客户识别程序（CIP）、客户尽职调查（CDD）、持续或定期监控。

1. 客户识别程序（CIP）

    客户识别程序（CIP）在入职过程或账户创建过程中收集信息（如姓名、出生日期和地址）。作为这一过程的一部分，组织需要在合理的时间范围内验证客户的身份。

    这个验证过程可以包括身份证件（ID）验证、面对面或当面验证、地址验证（例如，公用事业账单）、生物特征验证，或者这些的任何组合。

    KYC政策是基于风险评估策略来决定的。通常会考虑账户类型、提供的服务以及客户的地理位置等因素。

2. 客户尽职调查（CDD）

    客户尽职调查（CDD）是建立您的企业与您的客户之间信任关系的关键组成部分。根据关系中涉及的风险，有不同级别的客户尽职调查。

    简化的尽职调查适用于被认为存在低风险的欺诈或其他非法活动的情况。基础CDD是标准方法。而在高风险情况下，将采取增强型尽职调查。

3. 持续或定期监控

    持续或定期监控适用于初始检查不足以建立长期信任的情况，以及检查客户情况是否发生变化的情况。可能需要进行持续监控的情况包括：不寻常的账户活动（例如，突然增加）、欺诈或非法行为的增加，以及客户被列入制裁名单。监控的级别通常取决于基于风险的评估和策略。

## 人工智能语音识别用于KYC的部分

1. CIP阶段，客户识别过程可以通过语音异常检测模型，去检测是否合成声音，来鉴别伪造情况，并对重点、长期客户建立语音资料库，检测

2. CDD阶段，通过已有的语音资料检测人声

3. 持续更新语音资料库

## 语音检测用于KYC的优势

语音检测在KYC（Know Your Customer）流程中扮演着重要的角色，其主要作用可以从以下几个方面来理解：
- 身份验证：语音检测技术可以分析客户的语音特征，如音调、节奏、口音等，以此确认客户的身份。这种生物识别技术比传统的方法（如密码或安全问题）更难伪造，提高了身份验证的可靠性。
- 欺诈检测：通过分析语音的模式和特征，语音检测可以帮助金融机构识别出潜在的欺诈行为。例如，如果一个人的语音与之前记录的语音明显不同，系统可能会标记该交易进行进一步审查。
- 情感分析：在KYC流程中，通过语音检测进行的情感分析可以帮助识别客户是否感到紧张或不安，这可能表明他们可能在进行不诚实的交易或试图隐瞒信息。
- 非面对面交互：在远程开户或其他金融服务场景中，语音检测允许客户在不与银行工作人员面对面交流的情况下完成KYC流程，这对于提升用户体验和降低运营成本都很有帮助。
合规性：金融机构必须遵守反洗钱（AML）和反恐怖融资法规，语音检测作为一种强化身份验证的方法，有助于金融机构满足这些合规要求。
- 持续监控：在客户关系管理中，语音检测可用于持续监控交易行为，以便及时发现异常活动，这对于预防和减少金融犯罪至关重要。
- 交叉验证：语音检测可以与其他生物识别技术（如指纹、面部识别）结合使用，提供多层次的验证，从而进一步增强KYC流程的安全性。
。
