# Generated by Django 2.2.7 on 2020-02-19 06:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0023_remove_pln_text'),
    ]

    operations = [
        migrations.AddField(
            model_name='pln',
            name='text',
            field=models.TextField(default=1, max_length=700),
            preserve_default=False,
        ),
    ]
