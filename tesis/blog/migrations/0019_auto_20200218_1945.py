# Generated by Django 2.2.7 on 2020-02-19 00:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0018_auto_20200218_1944'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pln',
            name='text',
            field=models.TextField(max_length=700),
        ),
    ]