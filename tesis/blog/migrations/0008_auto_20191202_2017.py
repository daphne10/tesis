# Generated by Django 2.2.7 on 2019-12-03 01:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0007_auto_20191201_2212'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pln',
            name='image',
        ),
        migrations.AddField(
            model_name='pln',
            name='test',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='pln',
            name='image_result',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]
