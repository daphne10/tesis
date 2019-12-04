# Generated by Django 2.2.7 on 2019-11-28 08:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0004_pln_result'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pln',
            name='emotion',
            field=models.CharField(choices=[('alegria', 'Alegría'), ('confianza', 'Confianza'), ('miedo', 'Miedo'), ('sorpresa', 'Sorpresa'), ('tristeza', 'Tristeza'), ('aversion', 'Aversión'), ('anticipacion', 'Anticipación')], default='alegria', max_length=9),
        ),
    ]
