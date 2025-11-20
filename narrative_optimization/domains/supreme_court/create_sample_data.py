"""
Create Sample Supreme Court Dataset

Creates realistic sample data from famous Supreme Court cases
for demonstration and testing purposes.

This allows immediate testing of analysis pipeline while
full CourtListener API integration is set up.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import json
from pathlib import Path

# Famous Supreme Court cases with opinions (abbreviated but realistic)
SAMPLE_CASES = [
    {
        'case_id': 'brown-v-board-347-us-483',
        'year': 1954,
        'case_name': 'Brown v. Board of Education',
        'docket_number': '1',
        'date_filed': '1954-05-17',
        
        'majority_opinion': """
        We conclude that in the field of public education the doctrine of separate but equal has no place. 
        Separate educational facilities are inherently unequal. Therefore, we hold that the plaintiffs and 
        others similarly situated for whom the actions have been brought are, by reason of the segregation 
        complained of, deprived of the equal protection of the laws guaranteed by the Fourteenth Amendment.
        
        To separate children in grade and high schools from others of similar age and qualifications solely 
        because of their race generates a feeling of inferiority as to their status in the community that 
        may affect their hearts and minds in a way unlikely ever to be undone. Whatever may have been the 
        extent of psychological knowledge at the time of Plessy v. Ferguson, this finding is amply supported 
        by modern authority. Any language in Plessy contrary to this finding is rejected.
        
        We have now announced that such segregation is a denial of the equal protection of the laws. In order 
        that we may have the full assistance of the parties in formulating decrees, the cases will be restored 
        to the docket, and the parties are requested to present further argument.
        """,
        
        'dissenting_opinion': '',
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 8,
            'unanimous': True,
            'winner': 'petitioner',
            'citation_count': 45000,
            'precedent_setting': True,
            'overturned': False
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Warren, Earl',
            'author_id': 'warren-earl',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 267,
            'citation_count': 45000,
            'area_of_law': 'constitutional_civil_rights'
        }
    },
    
    {
        'case_id': 'roe-v-wade-410-us-113',
        'year': 1973,
        'case_name': 'Roe v. Wade',
        'docket_number': '70-18',
        'date_filed': '1973-01-22',
        
        'majority_opinion': """
        We forthwith acknowledge our awareness of the sensitive and emotional nature of the abortion controversy, 
        of the vigorous opposing views, even among physicians, and of the deep and seemingly absolute convictions 
        that the subject inspires. One's philosophy, one's experiences, one's exposure to the raw edges of human 
        existence, one's religious training, one's attitudes toward life and family and their values, and the 
        moral standards one establishes and seeks to observe, are all likely to influence and to color one's 
        thinking and conclusions about abortion.
        
        The Constitution does not explicitly mention any right of privacy. However, the Court has recognized that 
        a right of personal privacy, or a guarantee of certain areas or zones of privacy, does exist under the 
        Constitution. This right of privacy is broad enough to encompass a woman's decision whether or not to 
        terminate her pregnancy. The detriment that the State would impose upon the pregnant woman by denying 
        this choice altogether is apparent. Specific and direct harm medically diagnosable even in early pregnancy 
        may be involved. Maternity, or additional offspring, may force upon the woman a distressful life and 
        future. Psychological harm may be imminent. Mental and physical health may be taxed by child care.
        
        We therefore conclude that the right of personal privacy includes the abortion decision, but that this 
        right is not unqualified and must be considered against important state interests in regulation. Where 
        certain fundamental rights are involved, the Court has held that regulation limiting these rights may be 
        justified only by a compelling state interest. We repeat that this right is not absolute and is subject 
        to some limitations, and that at some point the state interests as to protection of health, medical 
        standards, and prenatal life, become dominant.
        """,
        
        'dissenting_opinion': """
        I find nothing in the language or history of the Constitution to support the Court's judgment. The Court 
        simply fashions and announces a new constitutional right for pregnant mothers and, with scarcely any 
        reason or authority for its action, invests that right with sufficient substance to override most existing 
        state abortion statutes. As an exercise of raw judicial power, the Court perhaps has authority to do what 
        it does today; but in my view its judgment is an improvident and extravagant exercise of the power of 
        judicial review that the Constitution extends to this Court.
        """,
        
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 5,
            'unanimous': False,
            'winner': 'petitioner',
            'citation_count': 38000,
            'precedent_setting': True,
            'overturned': True  # Dobbs v. Jackson (2022)
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Blackmun, Harry',
            'author_id': 'blackmun-harry',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 412,
            'citation_count': 38000,
            'area_of_law': 'constitutional_privacy'
        }
    },
    
    {
        'case_id': 'miranda-v-arizona-384-us-436',
        'year': 1966,
        'case_name': 'Miranda v. Arizona',
        'docket_number': '759',
        'date_filed': '1966-06-13',
        
        'majority_opinion': """
        The prosecution may not use statements, whether exculpatory or inculpatory, stemming from custodial 
        interrogation of the defendant unless it demonstrates the use of procedural safeguards effective to 
        secure the privilege against self-incrimination. By custodial interrogation, we mean questioning initiated 
        by law enforcement officers after a person has been taken into custody or otherwise deprived of his freedom 
        of action in any significant way.
        
        As for the procedural safeguards to be employed, unless other fully effective means are devised to inform 
        accused persons of their right of silence and to assure a continuous opportunity to exercise it, the 
        following measures are required. Prior to any questioning, the person must be warned that he has a right 
        to remain silent, that any statement he does make may be used as evidence against him, and that he has a 
        right to the presence of an attorney, either retained or appointed.
        
        The warning of the right to remain silent must be accompanied by the explanation that anything said can 
        and will be used against the individual in court. This warning is needed in order to make him aware not 
        only of the privilege, but also of the consequences of forgoing it. It is only through an awareness of 
        these consequences that there can be any assurance of real understanding and intelligent exercise of the 
        privilege.
        """,
        
        'dissenting_opinion': """
        The Court's new rules are not designed to guard against police brutality or other unmistakably banned 
        forms of coercion. Those who use third-degree tactics and deny them in court are equally able and destined 
        to lie as skillfully about warnings and waivers. Rather, the thrust of the new rules is to negate all 
        pressures, to reinforce the nervous or ignorant suspect, and ultimately to discourage any confession at 
        all. The rule announced today will measurably weaken the ability of the criminal law to perform these 
        tasks. It is a deliberate calculus to prevent interrogations, to reduce the incidence of confessions.
        """,
        
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 1,
            'unanimous': False,
            'winner': 'petitioner',
            'citation_count': 52000,
            'precedent_setting': True,
            'overturned': False
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Warren, Earl',
            'author_id': 'warren-earl',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 358,
            'citation_count': 52000,
            'area_of_law': 'criminal_procedure'
        }
    },
    
    # Add more landmark cases...
    {
        'case_id': 'citizens-united-558-us-310',
        'year': 2010,
        'case_name': 'Citizens United v. Federal Election Commission',
        'docket_number': '08-205',
        'date_filed': '2010-01-21',
        
        'majority_opinion': """
        If the First Amendment has any force, it prohibits Congress from fining or jailing citizens, or 
        associations of citizens, for simply engaging in political speech. The Government may regulate corporate 
        political speech through disclaimer and disclosure requirements, but it may not suppress that speech 
        altogether. We find no basis for the proposition that, in the context of political speech, the Government 
        may impose restrictions on certain disfavored speakers.
        
        Speech is an essential mechanism of democracy, for it is the means to hold officials accountable to the 
        people. The right of citizens to inquire, to hear, to speak, and to use information to reach consensus 
        is a precondition to enlightened self-government and a necessary means to protect it. For these reasons, 
        political speech must prevail against laws that would suppress it, whether by design or inadvertence.
        
        When Government seeks to use its full power, including the criminal law, to command where a person may 
        get his or her information or what distrusted source he or she may not hear, it uses censorship to 
        control thought. This is unlawful. The First Amendment confirms the freedom to think for ourselves.
        """,
        
        'dissenting_opinion': """
        The majority's approach to corporate electioneering marks a dramatic break from our past. The Court's 
        ruling threatens to undermine the integrity of elected institutions across the Nation. In the context 
        of election to public office, the distinction between corporate and human speakers is significant. 
        Corporations have no consciences, no beliefs, no feelings, no thoughts, no desires. They are not 
        themselves members of "We the People" by whom and for whom our Constitution was established.
        
        The majority's central argument is that the Government cannot restrict political speech based on the 
        speaker's corporate identity. But this misses the point. The rule against corporate expenditures is 
        not a complete ban on speech; it is a limitation on the way corporations can spend their money. The 
        law's distinction between corporate and individual political speech is eminently reasonable. The 
        majority today rejects a century of history when it treats the distinction as constitutionally suspect.
        """,
        
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 1,
            'unanimous': False,
            'winner': 'petitioner',
            'citation_count': 15000,
            'precedent_setting': True,
            'overturned': False
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Kennedy, Anthony',
            'author_id': 'kennedy-anthony',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 318,
            'citation_count': 15000,
            'area_of_law': 'first_amendment_corporate_speech'
        }
    },
    
    {
        'case_id': 'obergefell-v-hodges-576-us-644',
        'year': 2015,
        'case_name': 'Obergefell v. Hodges',
        'docket_number': '14-556',
        'date_filed': '2015-06-26',
        
        'majority_opinion': """
        The Constitution promises liberty to all within its reach, a liberty that includes certain specific 
        rights that allow persons, within a lawful realm, to define and express their identity. The petitioners 
        in these cases seek to find that liberty by marrying someone of the same sex and having their marriages 
        deemed lawful on the same terms and conditions as marriages between persons of the opposite sex.
        
        The history of marriage is one of both continuity and change. Changes, such as the decline of arranged 
        marriages and the abandonment of the law of coverture, have worked deep transformations in the structure 
        of marriage, affecting aspects of marriage once viewed as essential. These new insights have strengthened, 
        not weakened, the institution of marriage. Indeed, changed understandings of marriage are characteristic 
        of a Nation where new dimensions of freedom become apparent to new generations.
        
        The nature of injustice is that we may not always see it in our own times. The generations that wrote 
        and ratified the Bill of Rights and the Fourteenth Amendment did not presume to know the extent of 
        freedom in all of its dimensions, and so they entrusted to future generations a charter protecting the 
        right of all persons to enjoy liberty as we learn its meaning. When new insight reveals discord between 
        the Constitution's central protections and a received legal stricture, a claim to liberty must be 
        addressed. The dynamic of our constitutional system is that individuals need not await legislative action 
        before asserting a fundamental right.
        """,
        
        'dissenting_opinion': """
        The majority's decision represents the culmination of the latter tendency. Expanding required state 
        recognition of marriage to same-sex couples poses no threat to opposite-sex marriage, yet this Court 
        is not free to revise the Constitution in whatever way we might think useful. The fundamental right to 
        marry does not include a right to make a State change its definition of marriage. And a State's decision 
        to maintain the meaning of marriage that has persisted in every culture throughout human history can 
        hardly be called irrational.
        
        If you are among the many Americans—of whatever sexual orientation—who favor expanding same-sex marriage, 
        by all means celebrate today's decision. But do not celebrate the Constitution. It had nothing to do with 
        it. The majority's decision usurps the constitutional right of the people to decide this issue. Just who 
        do we think we are?
        """,
        
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 1,
            'unanimous': False,
            'winner': 'petitioner',
            'citation_count': 12000,
            'precedent_setting': True,
            'overturned': False
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Kennedy, Anthony',
            'author_id': 'kennedy-anthony',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 421,
            'citation_count': 12000,
            'area_of_law': 'constitutional_equal_protection'
        }
    },
    
    {
        'case_id': 'district-of-columbia-v-heller-554-us-570',
        'year': 2008,
        'case_name': 'District of Columbia v. Heller',
        'docket_number': '07-290',
        'date_filed': '2008-06-26',
        
        'majority_opinion': """
        The Second Amendment protects an individual right to possess a firearm unconnected with service in a 
        militia, and to use that arm for traditionally lawful purposes, such as self-defense within the home. 
        The Amendment's prefatory clause announces a purpose, but does not limit or expand the scope of the 
        second part, the operative clause. The operative clause's text and history demonstrate that it connotes 
        an individual right to keep and bear arms.
        
        The Constitution's text does not turn on the existence of a well regulated militia. The Second Amendment 
        protects "the right of the people," which is exercised individually and belongs to all Americans. The 
        prefatory clause does not suggest that preserving the militia was the only reason Americans valued the 
        ancient right; most undoubtedly thought it even more important for self-defense and hunting. But the 
        threat that the new Federal Government would destroy the citizens' militia by taking away their arms 
        was the reason that right was codified in a written Constitution.
        
        We are aware of the problem of handgun violence in this country, and we take seriously the concerns 
        raised by the many amici who believe that prohibition of handgun ownership is a solution. The Constitution 
        leaves the District of Columbia a variety of tools for combating that problem, including some measures 
        regulating handguns. But the enshrinement of constitutional rights necessarily takes certain policy 
        choices off the table.
        """,
        
        'dissenting_opinion': """
        The Second Amendment was adopted to protect the right of the people of each of the several States to 
        maintain a well-regulated militia. The Court today tries to impose a new interpretation on the Amendment, 
        but the text and history do not support it. The Amendment's text includes introductory language about 
        the importance of a well regulated Militia for the security of a free State. That language matters.
        
        The majority's conclusion is wrong for two independent reasons. First, that conclusion is inconsistent 
        with the text and history of the Amendment. Second, even if the Amendment did protect an individual 
        right, that right was limited to keeping and bearing arms for military purposes, not self-defense.
        """,
        
        'concurring_opinion': '',
        'plurality_opinion': '',
        'opinion_full_text': '',
        'petitioner_brief': '',
        'respondent_brief': '',
        'oral_arguments': '',
        
        'outcome': {
            'vote_margin': 1,
            'unanimous': False,
            'winner': 'respondent',
            'citation_count': 18000,
            'precedent_setting': True,
            'overturned': False
        },
        
        'metadata': {
            'court': 'scotus',
            'author': 'Scalia, Antonin',
            'author_id': 'scalia-antonin',
            'per_curiam': False,
            'opinion_type': 'majority',
            'download_url': '',
            'absolute_url': '',
            'word_count': 398,
            'citation_count': 18000,
            'area_of_law': 'second_amendment'
        }
    }
]


def create_extended_sample():
    """Create extended sample with more cases (including non-landmark)."""
    
    # Add routine cases (unanimous, lower citations)
    routine_cases = []
    
    for i in range(20):
        routine_case = {
            'case_id': f'routine-case-{i}',
            'year': 2010 + (i % 13),
            'case_name': f'United States v. Defendant {i}',
            'docket_number': f'10-{1000+i}',
            'date_filed': f'{2010 + (i % 13)}-03-15',
            
            'majority_opinion': f"""
            The question presented is whether the lower court erred in its interpretation of the statute.
            We hold that it did not. The statutory language is clear and unambiguous. When Congress enacted
            this provision, it intended to cover situations like this one. The legislative history supports
            this interpretation. The practical consequences of the alternative interpretation would be
            unworkable and contrary to the statute's purpose.
            
            The petitioner argues that we should adopt a different reading, but this argument is unpersuasive.
            The text does not support it, and our precedents counsel against it. In prior cases, we have
            consistently interpreted similar language in the same way. See United States v. Prior Case (2005).
            
            Therefore, we affirm the judgment of the Court of Appeals. The statute applies as the lower court
            held, and the petitioner's claim fails. This interpretation is consistent with the statutory
            scheme and congressional intent. Justice Smith concurs. Case {i} decided.
            """,
            
            'dissenting_opinion': '',
            'concurring_opinion': '',
            'plurality_opinion': '',
            'opinion_full_text': '',
            'petitioner_brief': '',
            'respondent_brief': '',
            'oral_arguments': '',
            
            'outcome': {
                'vote_margin': 8,
                'unanimous': True,
                'winner': 'respondent',
                'citation_count': 50 + (i * 20),  # Routine cases: low citations
                'precedent_setting': False,
                'overturned': False
            },
            
            'metadata': {
                'court': 'scotus',
                'author': 'Roberts, John',
                'author_id': 'roberts-john',
                'per_curiam': False,
                'opinion_type': 'majority',
                'download_url': '',
                'absolute_url': '',
                'word_count': 250,
                'citation_count': 50 + (i * 20),
                'area_of_law': 'statutory_interpretation'
            }
        }
        routine_cases.append(routine_case)
    
    return SAMPLE_CASES + routine_cases


def main():
    """Create sample dataset."""
    print("Creating Supreme Court sample dataset...")
    
    # Create extended sample
    all_cases = create_extended_sample()
    
    # Save to data directory
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'supreme_court_complete.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_cases, f, indent=2)
    
    print(f"✅ Created sample dataset: {len(all_cases)} cases")
    print(f"   - Landmark cases: {len(SAMPLE_CASES)}")
    print(f"   - Routine cases: {len(all_cases) - len(SAMPLE_CASES)}")
    print(f"   - Unanimous: {sum(1 for c in all_cases if c['outcome']['unanimous'])}")
    print(f"   - Split: {sum(1 for c in all_cases if not c['outcome']['unanimous'])}")
    print(f"\n✅ Saved to: {output_path}")
    print(f"\nNext step: Run analysis")
    print(f"python3 narrative_optimization/domains/supreme_court/analyze_complete.py")


if __name__ == '__main__':
    main()

