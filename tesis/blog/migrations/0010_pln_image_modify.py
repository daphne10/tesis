# Generated by Django 2.2.7 on 2019-12-03 10:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0009_auto_20191202_2044'),
    ]

    operations = [
        migrations.AddField(
            model_name='pln',
            name='image_modify',
            field=models.ImageField(blank=True, null=True, upload_to='images/'),
        ),
    ]
